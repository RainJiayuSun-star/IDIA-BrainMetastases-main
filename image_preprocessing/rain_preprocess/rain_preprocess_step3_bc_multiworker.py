#!/usr/bin/env python3
"""Parallel CPU N4: one scan per task, exclusive per-output OS locks.

Windows, Linux and macOS: python3 rain_preprocess_step3_bc_multiworker.py
    -i /path/to/exams -o /path/to/corrected -w 4

Requires SimpleITK and numpy. No shell commands or GPU packages are needed.
YAML mode additionally requires PyYAML:
    python3 rain_preprocess_step3_bc_multiworker.py --config config_step3.example.yaml --dry-run
Without an output directory, corrected images are saved beside originals.
Use --sequences t1c t1n t2f t2w for UCSD anatomical scans. Discovery is one
folder level below input_root. Without --sequences all non-label scans are used.

Locks and completion records persist centrally under <output_root>/.n4_logs.
Never unlink central lock files while jobs may run.
The OS releases the lock when its owning process closes it or exits. All runs
targeting the same output tree must use this locking protocol. Use a local
filesystem, or a shared filesystem with working cross-client advisory locks.
Independent output trees are independent jobs. Do not run an older script
against the same output directory concurrently.
"""

import argparse
import concurrent.futures
import contextlib
import csv
import errno
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import string
import sys
import time
import uuid

import numpy as np
import SimpleITK as sitk

if os.name == "nt":
    import msvcrt
else:
    import fcntl

DATASETS_CONFIG = {
    "1": {
        "name": "Brain-Mets-Lung",
        "input_root": "/Users/rains/rainworkspace/2-IDIALab/Datasets/PKG_Brain-Mets-Lung-MRI-Path-Segs_radiology_images/Brain-Mets-Lung-MRI-Path-Segs",
        "output_root": "/Users/rains/rainworkspace/2-IDIALab/Datasets/PKG_Brain-Mets-Lung-MRI-Path-Segs_radiology_images/Brain-Mets-Lung_MRI_BC_Output",
    },
    "2": {
        "name": "BraTS2025-MET",
        "input_root": r"D:\A1_RainSun_20240916\1-UWMadison\IDiA-Lab\Medical_Images_Public\BraTS2026\MICCAI-LH-BraTS2025-MET-Challenge-Training",
        "output_root": r"D:\A1_RainSun_20240916\1-UWMadison\IDiA-Lab\Medical_Images_Public\BraTS2026\BraTS2025-MET-Training-BC_complete",
    },
}
DEFAULT_PARAMETERS = dict(shrink_factor=2, num_fitting_levels=4, num_iters=50,
                          convergence_thresh=1e-6)
OUTPUT_SUFFIX = "_BC.nii.gz"
ALGORITHM_VERSION = "sitk-n4-scan-worker-v1"


class ScanBusy(Exception):
    """Another process owns this output's lock."""


@contextlib.contextmanager
def output_lock(output):
    """Serialize all updated invocations by canonical output identity."""
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with kernel_lock(bookkeeping_path(output, '.lock')):
        yield


@contextlib.contextmanager
def kernel_lock(path):
    """Nonblocking kernel lock; file age never permits stealing a live lock.

    Keep the lock inode/path permanently. Removing it can let a third process
    lock a new inode while a previous waiter still holds the old inode.
    Unexpected filesystem/permission errors fail closed, not as 'unlocked'.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as handle:
        # Windows locks one byte. An append-only initialization is harmless
        # even if another opener concurrently initializes the first byte.
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise ScanBusy(str(path)) from exc
            raise
        try:
            yield handle
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def canonical(path):
    return os.path.normcase(str(Path(path).expanduser().resolve()))


def file_identity(path):
    p = Path(path).resolve()
    stat = p.stat()
    return dict(path=canonical(p), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def output_name(name):
    if name.lower().endswith(".nii.gz"):
        return name[:-7] + OUTPUT_SUFFIX
    if name.lower().endswith(".nii"):
        return name[:-4] + OUTPUT_SUFFIX
    raise ValueError("Not a NIfTI scan: " + name)


def scan_signature(task, params, itk_threads_per_worker=1):
    # Stat provenance is fast; published output additionally has a SHA-256.
    # Inputs must remain immutable during a run.
    return dict(algorithm=ALGORITHM_VERSION, simpleitk=sitk.Version_VersionString(),
                source=file_identity(task['input']),
                mask=file_identity(task['mask']) if task.get('mask') else None,
                parameters=params, itk_threads_per_worker=itk_threads_per_worker)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def same_grid(a, b):
    return (a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing(), rtol=0, atol=1e-6)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), rtol=0, atol=1e-5)
            and np.allclose(a.GetDirection(), b.GetDirection(), rtol=0, atol=1e-6))


def validate_image(path, reference):
    image = sitk.ReadImage(str(path))  # Full read, not just a nonempty file.
    if not same_grid(image, reference):
        raise ValueError("Output grid differs from source: " + str(path))
    if image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Expected a scalar image: " + str(path))
    if not np.isfinite(sitk.GetArrayViewFromImage(image)).all():
        raise ValueError("Nonfinite output: " + str(path))


def completion_record(output):
    return bookkeeping_path(output, '.complete.json')


def bookkeeping_path(output, suffix):
    # Outputs always live in <root>/<case>/<filename>. Resolve first so aliases
    # share both the same root and the same stable path-derived identity.
    output = Path(canonical(output))
    key = hashlib.sha256(str(output).encode('utf-8')).hexdigest()
    return output.parent.parent / '.n4_logs' / 'outputs' / (key + suffix)


def existing_completion_record(output):
    central = completion_record(output)
    return central if central.exists() else Path(str(output) + '.complete.json')


def is_complete(task, params, reference=None, itk_threads_per_worker=1):
    """Require matching provenance, decoded geometry/values and output hash."""
    try:
        output = Path(task['output'])
        record = json.loads(existing_completion_record(output).read_text(encoding='utf-8'))
        signature = dict(record['signature'])
        # Pre-thread-setting completion records were produced with one ITK thread.
        signature.setdefault('itk_threads_per_worker', 1)
        if signature != scan_signature(task, params, itk_threads_per_worker):
            return False
        if record['output_sha256'] != sha256(output):
            return False
        if reference is None:
            reference = sitk.ReadImage(task['input'])
        validate_image(output, reference)
        return True
    except (OSError, ValueError, KeyError, RuntimeError):
        return False


def temporary_path(output, suffix):
    output = Path(output)
    return output.parent / ('.' + output.name + '.' + uuid.uuid4().hex + suffix)


def atomic_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path, '.tmp')
    try:
        with open(temporary, 'x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def initialize_worker(itk_threads_per_worker=1):
    if type(itk_threads_per_worker) is not int or itk_threads_per_worker < 1:
        raise ValueError('itk_threads_per_worker must be an integer >= 1')
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(itk_threads_per_worker)
    effective = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    inherited = sitk.N4BiasFieldCorrectionImageFilter().GetNumberOfThreads()
    if inherited != effective:
        raise RuntimeError('New N4 filter did not inherit the effective ITK thread setting')
    if effective != itk_threads_per_worker:
        print(f'[Warning] ITK requested {itk_threads_per_worker} threads per worker; '
              f'effective limit is {effective}.', flush=True)
    return dict(itk_threads_requested=itk_threads_per_worker, itk_threads_effective=effective,
                n4_threads_effective=inherited, itk_threader=sitk.ProcessObject.GetGlobalDefaultThreader())


def correct_scan(task, params, itk_threads_per_worker=1):
    """One input image only; caller must own its output lock."""
    image = sitk.ReadImage(task['input'], sitk.sitkFloat32)
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Only scalar 3D scans are supported")
    if not np.isfinite(sitk.GetArrayViewFromImage(image)).all():
        raise ValueError("Input contains nonfinite values")
    if task.get('mask'):
        mask = sitk.ReadImage(task['mask'])
        if not same_grid(mask, image):
            raise ValueError("Brain mask must match the scan's physical grid")
        mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
    else:
        mask = sitk.OtsuThreshold(image, 0, 1)
    # Log-intensity fitting requires positive input; preserve the original
    # application formula outside the estimation mask.
    mask = sitk.Cast((mask > 0) & (image > 0), sitk.sitkUInt8)
    shrink = params['shrink_factor']
    small_image = sitk.Shrink(image, [shrink] * 3) if shrink > 1 else image
    small_mask = sitk.Shrink(mask, [shrink] * 3) if shrink > 1 else mask
    if not np.any(sitk.GetArrayViewFromImage(small_mask)):
        raise ValueError("Empty positive-signal estimation mask after shrinking")
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    effective_threads = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    if corrector.GetNumberOfThreads() != effective_threads:
        raise RuntimeError('N4 filter thread limit differs from the configured ITK default')
    corrector.SetMaximumNumberOfIterations([params['num_iters']] * params['num_fitting_levels'])
    corrector.SetConvergenceThreshold(params['convergence_thresh'])
    corrector.Execute(small_image, small_mask)
    corrected = image / sitk.Exp(corrector.GetLogBiasFieldAsImage(image))
    output = Path(task['output'])
    signature = scan_signature(task, params, itk_threads_per_worker)
    temporary = temporary_path(output, '.nii.gz')
    try:
        sitk.WriteImage(corrected, str(temporary))
        validate_image(temporary, image)
        # Do not commit if the source/mask changed during processing.
        if signature != task['_initial_signature']:
            raise ValueError("Source or mask changed while processing")
        digest = sha256(temporary)
        os.replace(temporary, output)
        execution = dict(itk_threads_requested=itk_threads_per_worker,
                         itk_threads_effective=effective_threads,
                         n4_threads_effective=corrector.GetNumberOfThreads(),
                         itk_threader=sitk.ProcessObject.GetGlobalDefaultThreader())
        atomic_json(completion_record(output), dict(signature=signature, output_sha256=digest,
                                                   execution=execution))
    finally:
        temporary.unlink(missing_ok=True)


def process_scan_worker(task, params, itk_threads_per_worker=1):
    """Exactly one scan per task, locked across both pools and independent runs."""
    execution = initialize_worker(itk_threads_per_worker)
    start = time.monotonic()
    result = dict(patient=task['patient'], scan=task['input'], output=task['output'],
                  worker_pid=os.getpid(), status='error', elapsed=0.0, error=None, **execution)
    try:
        with output_lock(task['output']):
            # Completion check MUST happen after locking: a previous owner may
            # have published a valid output while this task was being queued.
            if is_complete(task, params, itk_threads_per_worker=itk_threads_per_worker):
                result['status'] = 'skipped'
            else:
                task = dict(task, _initial_signature=scan_signature(task, params, itk_threads_per_worker))
                correct_scan(task, params, itk_threads_per_worker)
                result['status'] = 'done'
    except ScanBusy:
        result['status'] = 'locked'
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    result['elapsed'] = time.monotonic() - start
    return result


def build_scan_tasks(patient_entries):
    """Deduplicate identical destinations; reject conflicting/aliased mappings."""
    tasks = {}
    input_outputs = {}
    for entry in patient_entries:
        for source in entry['structural_scans']:
            output = str(Path(entry['out_dir']) / output_name(Path(source).name))
            task = dict(patient=entry['id'], input=str(Path(source).resolve()),
                        output=str(Path(output).resolve()),
                        mask=(entry.get('mask_by_scan') or {}).get(Path(source).name) or entry.get('mask_path'))
            if task['mask']:
                task['mask'] = str(Path(task['mask']).resolve())
            key = canonical(output)
            if canonical(source) == key:
                raise ValueError("Input/output overlap: " + output)
            if key in tasks and tasks[key] != task:
                raise ValueError("Different inputs or masks map to the same output: " + output)
            source_key = canonical(source)
            if source_key in input_outputs and input_outputs[source_key] != key:
                raise ValueError("Same input scan mapped to multiple outputs: " + source)
            tasks[key] = task
            input_outputs[source_key] = key
    return list(tasks.values())


def copy_auxiliary_files(patient_entries):
    """Master-only copying, serialized across independent invocations too."""
    for entry in patient_entries:
        for source in entry.get('other_files', []):
            target = Path(entry['out_dir']) / Path(source).name
            if canonical(source) == canonical(target):
                continue  # In-place mode: companions are already present.
            try:
                with output_lock(target):
                    if target.exists() and sha256(source) == sha256(target):
                        continue
                    temporary = temporary_path(target, '.tmp')
                    try:
                        shutil.copyfile(source, temporary)
                        os.replace(temporary, target)
                    finally:
                        temporary.unlink(missing_ok=True)
            except ScanBusy:
                # Fail visibly so this run cannot claim all files are present.
                raise RuntimeError("Auxiliary copy is busy; rerun to verify: " + str(target))


def export_reports(csv_path):
    """Optional legacy Excel/PDF reports, created only by the master process."""
    with open(csv_path, newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    try:
        import pandas as pd
        pd.DataFrame(rows).to_excel(Path(csv_path).with_suffix('.xlsx'), index=False)
    except (ImportError, OSError, ValueError) as exc:
        print(f'Excel report unavailable ({exc}); full CSV log is saved.', flush=True)
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Courier', size=8)
        for row in rows:
            text = f"{row['status']} | {row['scan']} | {row['elapsed']}s | {row['error']}"
            text += (f" | ITK requested={row.get('itk_threads_requested', '1')}"
                     f" effective={row.get('itk_threads_effective', 'unknown')}"
                     f" N4={row.get('n4_threads_effective', 'unknown')}"
                     f" threader={row.get('itk_threader', 'unknown')}")
            # Bound line length for both pyfpdf and fpdf2.
            text = text.encode('latin-1', 'replace').decode('latin-1')
            for offset in range(0, len(text), 95):
                pdf.cell(0, 4, text[offset:offset + 95], ln=1)
        pdf.output(str(Path(csv_path).with_suffix('.pdf')))
    except Exception as exc:
        print(f'PDF report unavailable ({exc}); full CSV log is saved.', flush=True)


def sync_completed_case(entry, completed_outputs, copied_cases):
    """Publish companions as soon as this case's selected scans are complete."""
    case_key = canonical(entry['out_dir'])
    required = {canonical(Path(entry['out_dir']) / output_name(Path(source).name))
                for source in entry['structural_scans']}
    if case_key in copied_cases or not required.issubset(completed_outputs):
        return False
    copy_auxiliary_files([entry])
    copied_cases.add(case_key)
    return True


def migrate_legacy_bookkeeping(tasks, entries, params, itk_threads_per_worker=1):
    """Successful-run cleanup only. Older script versions MUST be stopped.

    Central locks remain stable forever. Adjacent locks belong to the retired
    protocol: checking them catches active holders, but cannot detect queued
    older invocations. Never mix the two script versions during migration.
    """
    targets = {canonical(task['output']): task for task in tasks}
    for entry in entries:
        for source in entry.get('other_files', []):
            target = Path(entry['out_dir']) / Path(source).name
            if canonical(source) != canonical(target):
                targets.setdefault(canonical(target), None)
    for output, task in targets.items():
        legacy_lock = Path(output + '.lock')
        with output_lock(output), contextlib.ExitStack() as stack:
            if legacy_lock.exists():
                legacy_handle = stack.enter_context(kernel_lock(legacy_lock))
                legacy_handle.seek(0)
                if legacy_handle.read() != b'\0':
                    raise ValueError('Unrecognized legacy lock; preserved: ' + str(legacy_lock))
            if task is not None:
                if not is_complete(task, params, itk_threads_per_worker=itk_threads_per_worker):
                    raise RuntimeError('Output changed before bookkeeping migration: ' + output)
                legacy = Path(output + '.complete.json')
                if legacy.exists():
                    record = json.loads(legacy.read_text(encoding='utf-8'))
                    if (not isinstance(record, dict)
                            or not isinstance(record.get('signature'), dict)
                            or record.get('signature', {}).get('algorithm') != ALGORITHM_VERSION
                            or not isinstance(record.get('output_sha256'), str)):
                        raise ValueError('Unrecognized legacy completion record; preserved: ' + str(legacy))
                    if not completion_record(output).exists():
                        atomic_json(completion_record(output), record)
                    # Confirm publication before retiring the old resume record.
                    if not is_complete(task, params, itk_threads_per_worker=itk_threads_per_worker):
                        raise RuntimeError('Migrated completion record failed verification: ' + output)
                    legacy.unlink()
            # ExitStack releases the retired Windows byte lock before unlink.
            stack.close()
            if legacy_lock.exists():
                legacy_lock.unlink()


def logging_loop(operation_name, input_root, output_root, patient_entries, params,
                 num_workers=16, max_retries=3, retry_delay=2.0, selection_report=None,
                 itk_threads_per_worker=1):
    tasks = build_scan_tasks(patient_entries)
    root = Path(output_root) / '.n4_logs'
    root.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
    if selection_report is not None:
        write_selection_report(root, run_id, selection_report)
    log_path = root / (operation_name + '_' + run_id + '.csv')
    print(f'{platform.system()} | SimpleITK {sitk.Version_VersionString()} | spawn workers={num_workers} | '
          f'ITK threads/worker={itk_threads_per_worker} | requested computation-thread budget='
          f'{num_workers * itk_threads_per_worker}')
    print(f'Queued {len(tasks)} unique scans in {len(patient_entries)} folders; one scan per task.')
    print(f'Per-scan status log: {log_path}', flush=True)
    remaining = tasks
    completed_outputs = set()
    copied_cases = set()
    entry_by_output = {canonical(Path(entry['out_dir']) / output_name(Path(source).name)): entry
                       for entry in patient_entries for source in entry['structural_scans']}
    with open(log_path, 'x', newline='', encoding='utf-8') as stream:
        fields = ['pass', 'patient', 'scan', 'output', 'worker_pid', 'status', 'elapsed', 'error',
                  'itk_threads_requested', 'itk_threads_effective', 'n4_threads_effective', 'itk_threader']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for attempt in range(max_retries + 1):
            if attempt:
                time.sleep(retry_delay)
            failed = []
            # Spawn avoids fork-after-ITK initialization on Linux and is also
            # supported by Windows/macOS. Functions are module-level/picklable.
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=num_workers, mp_context=multiprocessing.get_context('spawn'),
                    initializer=initialize_worker, initargs=(itk_threads_per_worker,)) as pool:
                futures = {pool.submit(process_scan_worker, task, params, itk_threads_per_worker): task
                           for task in remaining}
                for future in concurrent.futures.as_completed(futures):
                    task = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = dict(patient=task['patient'], scan=task['input'], output=task['output'],
                                      worker_pid='', status='error', elapsed=0, error=str(exc),
                                      itk_threads_requested=itk_threads_per_worker,
                                      itk_threads_effective='', n4_threads_effective='', itk_threader='')
                    writer.writerow(dict(result, **{'pass': attempt + 1}))
                    stream.flush()
                    print(f"[{result['status']}] {task['patient']}/{Path(task['input']).name} "
                          f"({result['elapsed']:.1f}s) {result['error'] or ''}", flush=True)
                    if result['status'] in ('done', 'skipped'):
                        key = canonical(task['output'])
                        completed_outputs.add(key)
                        entry = entry_by_output[key]
                        if sync_completed_case(entry, completed_outputs, copied_cases) and entry.get('other_files'):
                            print(f"[companions ready] {entry['id']}", flush=True)
                    if result['status'] not in ('done', 'skipped'):
                        failed.append(task)
            remaining = failed
            if not remaining:
                break
    export_reports(log_path)
    # Verify all records after the final pass; locked scans remain pending.
    incomplete = [task for task in tasks if not is_complete(task, params,
                                                         itk_threads_per_worker=itk_threads_per_worker)]
    if incomplete:
        pending_path = root / ('pending_' + run_id + '.json')
        atomic_json(pending_path, incomplete)
        print(f'{len(incomplete)} scans remain incomplete or owned by another run: {pending_path}', flush=True)
        return False
    copy_auxiliary_files(patient_entries)
    migrate_legacy_bookkeeping(tasks, patient_entries, params, itk_threads_per_worker)
    print(f'Verified {len(tasks)} completed scans.', flush=True)
    return True


def is_nifti(path):
    return path.name.lower().endswith(('.nii.gz', '.nii'))


def discover_patient_files(input_root, output_root, patient_subset=None, sequences=None):
    root = Path(input_root)
    if not root.is_dir():
        raise ValueError('Input directory does not exist: ' + str(root))
    folders = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.'))
    if patient_subset:
        try:
            parts = patient_subset.split(':')
            if len(parts) != 2:
                raise ValueError()
            start, end = map(int, parts)
            if start < 0 or end <= start:
                raise ValueError()
            folders = folders[start:end]
        except ValueError:
            raise ValueError("Subset must be a zero-based start:end range, e.g. 0:5")
    entries = []
    for folder in folders:
        files = sorted(p for p in folder.iterdir() if p.is_file() and is_nifti(p) and not is_generated(p.name))
        masks = [p for p in files if any(k in p.name.lower() for k in ('brain_mask', '_bet.', '_ss_bet.'))]
        labels = [p for p in files if p not in masks and any(k in p.name.lower() for k in ('seg', 'label', 'tumor', 'mask'))]
        structural = [p for p in files if p not in masks + labels and not is_generated(p.name)]
        if sequences:
            allowed = {s.lower() for s in sequences}
            structural = [p for p in structural if (p.name[:-7] if p.name.lower().endswith('.nii.gz') else p.stem).rsplit('-', 1)[-1].lower() in allowed]
        if not structural:
            continue
        paired = {}
        for p in structural:
            stem = p.name[:-7] if p.name.lower().endswith('.nii.gz') else p.stem
            candidates = {stem + '_bet', stem + '_brain_mask', stem + '_SS_bet'}
            if stem.endswith('_SS'):
                candidates.add(stem + '_bet')
            match = [m for m in masks if (m.name[:-7] if m.name.lower().endswith('.nii.gz') else m.stem) in candidates]
            if len(match) == 1:
                paired[p.name] = str(match[0])
        if len(masks) > 1 and any(p.name not in paired for p in structural):
            raise ValueError('Ambiguous brain masks in ' + str(folder))
        entries.append(dict(id=folder.name, structural_scans=[str(p) for p in structural],
                            mask_path=str(masks[0]) if len(masks) == 1 else None,
                            mask_by_scan=paired, other_files=[str(p) for p in labels + masks],
                            out_dir=str(Path(output_root) / folder.name)))
    return entries


def is_generated(name):
    """Never rediscover our outputs, sidecars, locks, logs or temporary files."""
    low = name.lower()
    return (name.startswith('.') or low.endswith(('_bc.nii.gz', '_bc.nii',
                                                '.lock', '.complete.json', '.tmp')))


def check_keys(value, allowed, label):
    if not isinstance(value, dict):
        raise ValueError(label + ' must be a mapping')
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f'Unknown {label} keys: ' + ', '.join(sorted(map(str, unknown))))


def validate_template(template, label, image=False):
    if not isinstance(template, str) or not template:
        raise ValueError(label + ' must be a nonempty filename template')
    try:
        for _, field, spec, conversion in string.Formatter().parse(template):
            if field is not None and (field != 'case_id' or spec or conversion):
                raise ValueError('Only {case_id} is supported')
        rendered = template.format(case_id='example')
    except (ValueError, KeyError, IndexError) as exc:
        raise ValueError(f'Invalid {label}: {exc}') from exc
    if (rendered in ('.', '..') or any(c in rendered for c in '/\\:*?[]<>|{}')
            or is_generated(rendered)):
        raise ValueError(label + ' must be an original, immediate filename without wildcards or paths')
    if image and not rendered.lower().endswith(('.nii.gz', '.nii')):
        raise ValueError(label + ' must identify a .nii or .nii.gz image')
    return template


def resolve_path(value, base, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + ' must be a nonempty path')
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base / path).resolve())


def load_config(path):
    try:
        import yaml
    except ImportError as exc:
        raise ValueError('YAML mode requires PyYAML: python3 -m pip install PyYAML') from exc
    path = Path(path).expanduser().resolve()
    try:
        cfg = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    except yaml.YAMLError as exc:
        raise ValueError('Invalid YAML: ' + str(exc)) from exc
    check_keys(cfg, ('input_dir', 'output_dir', 'case_list', 'images', 'brain_mask_filename',
                     'companions', 'num_workers', 'itk_threads_per_worker', 'max_retries',
                     'retry_delay', 'n4_params'), 'config')
    cfg = dict(cfg)
    for key in ('input_dir', 'output_dir'):
        if cfg.get(key) is not None:
            cfg[key] = resolve_path(cfg[key], path.parent, key)
    images = cfg.get('images')
    if not isinstance(images, dict) or not images:
        raise ValueError('images must be a nonempty mapping of image types to filename rules')
    shared = cfg.get('brain_mask_filename')
    if shared is not None:
        validate_template(shared, 'brain_mask_filename', image=True)
    for name, rule in images.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError('Image type names must be nonempty strings')
        check_keys(rule, ('filename', 'mask_filename'), 'image ' + name)
        validate_template(rule.get('filename'), name + '.filename', image=True)
        if rule.get('mask_filename') is not None:
            validate_template(rule['mask_filename'], name + '.mask_filename', image=True)
    case_list = cfg.get('case_list', {})
    check_keys(case_list, ('path', 'id_column'), 'case_list')
    case_list = dict(case_list)
    if case_list.get('path') is not None:
        case_list['path'] = resolve_path(case_list['path'], path.parent, 'case_list.path')
    column = case_list.get('id_column', 'case_id')
    if not isinstance(column, str) or not column.strip():
        raise ValueError('case_list.id_column must be a nonempty string')
    case_list['id_column'] = column
    cfg['case_list'] = case_list
    companions = cfg.get('companions', {})
    check_keys(companions, ('mode', 'filenames'), 'companions')
    companions = dict(companions)
    companions.setdefault('mode', 'none')
    companions.setdefault('filenames', [])
    if companions['mode'] not in ('none', 'listed', 'all_unselected'):
        raise ValueError('companions.mode must be none, listed, or all_unselected')
    if not isinstance(companions['filenames'], list):
        raise ValueError('companions.filenames must be a list')
    for template in companions['filenames']:
        validate_template(template, 'companions.filenames')
    cfg['companions'] = companions
    check_keys(cfg.get('n4_params', {}), DEFAULT_PARAMETERS, 'n4_params')
    return cfg


def read_case_ids(settings):
    path = settings.get('path')
    if path is None:
        return None
    path = Path(path)
    with open(path, newline='', encoding='utf-8-sig') as stream:
        if path.suffix.lower() == '.csv':
            reader = csv.DictReader(stream)
            column = settings['id_column']
            if not reader.fieldnames or column not in reader.fieldnames:
                raise ValueError(f'Case CSV must contain column {column!r}')
            values = [row[column] for row in reader]
        elif path.suffix.lower() == '.txt':
            values = [line for line in stream if not line.lstrip().startswith('#')]
        else:
            raise ValueError('Case list must be a .csv or .txt file')
    ids = []
    for value in values:
        value = (value or '').strip()
        if not value:
            continue
        if value in ('.', '..') or any(c in value for c in '/\\:'):
            raise ValueError('Case list entries must be folder IDs, not paths: ' + value)
        if value not in ids:
            ids.append(value)
    return ids


def select_configured_cases(input_root, output_root, cfg, patient_subset=None):
    """Resolve exact names without opening image payloads or creating files."""
    root = Path(input_root)
    if not root.is_dir():
        raise ValueError('Input directory does not exist: ' + str(root))
    folders = {p.name: p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith('.')}
    ids = read_case_ids(cfg['case_list'])
    ids = sorted(folders) if ids is None else ids
    if patient_subset:
        try:
            start, end = map(int, patient_subset.split(':'))
            if start < 0 or end <= start:
                raise ValueError()
            ids = ids[start:end]
        except ValueError:
            raise ValueError('Subset must be a zero-based start:end range, e.g. 0:5')
    report = dict(considered_folders=len(folders), selected_folders=len(ids), eligible_folders=0,
                  skipped_folders=0, matched_images={name: 0 for name in cfg['images']},
                  eligible_images={name: 0 for name in cfg['images']}, records=[])
    entries = []
    in_place = canonical(input_root) == canonical(output_root)
    for case_id in ids:
        folder = folders.get(case_id)
        if folder is None:
            report['records'].append(dict(case_id=case_id, status='missing_folder', details='Listed folder is absent'))
            continue
        # Dictionary lookup makes exact template matching case-sensitive on all OSes.
        files = {p.name: p for p in folder.iterdir() if p.is_file() and not is_generated(p.name)}
        scans, paired, missing, resolved_names = [], {}, [], set()
        mask_names = set()
        for image_type, rule in cfg['images'].items():
            name = rule['filename'].format(case_id=case_id)
            validate_template(name, 'resolved image filename', image=True)
            if name in resolved_names:
                raise ValueError(f'{case_id}: multiple image types select the same file {name}')
            resolved_names.add(name)
            if name not in files:
                missing.append(name)
            else:
                scans.append(str(files[name]))
                report['matched_images'][image_type] += 1
            mask_rule = rule.get('mask_filename', cfg.get('brain_mask_filename'))
            if mask_rule is not None:
                mask_name = mask_rule.format(case_id=case_id)
                validate_template(mask_name, 'resolved mask filename', image=True)
                mask_names.add(mask_name)
                if mask_name not in files:
                    missing.append(mask_name)
                else:
                    paired[name] = str(files[mask_name])
        if resolved_names & mask_names:
            raise ValueError(f'{case_id}: an image selected for correction is also configured as a brain mask')
        if missing:
            report['records'].append(dict(case_id=case_id, status='skipped_missing_files',
                                          details='; '.join(dict.fromkeys(missing))))
            continue
        other = []
        companions = cfg['companions']
        if not in_place and companions['mode'] == 'listed':
            for template in companions['filenames']:
                name = template.format(case_id=case_id)
                validate_template(name, 'resolved companion filename')
                if name not in files:
                    report['records'].append(dict(case_id=case_id, status='missing_companion', details=name))
                elif name not in resolved_names:
                    other.append(str(files[name]))
        elif not in_place and companions['mode'] == 'all_unselected':
            other = [str(p) for name, p in sorted(files.items()) if name not in resolved_names]
        entries.append(dict(id=case_id, structural_scans=scans, mask_path=None, mask_by_scan=paired,
                            other_files=list(dict.fromkeys(other)), out_dir=str(Path(output_root) / case_id)))
        report['records'].append(dict(case_id=case_id, status='eligible', details=''))
        for name in cfg['images']:
            report['eligible_images'][name] += 1
    report['eligible_folders'] = len(entries)
    report['skipped_folders'] = len(ids) - len(entries)
    return entries, report


def validate_destinations(entries):
    tasks = build_scan_tasks(entries)
    # A companion must never replace a correction destination or an original.
    outputs = {canonical(task['output']) for task in tasks}
    sources = {canonical(source) for e in entries for source in e['structural_scans'] + e.get('other_files', [])}
    for task in tasks:
        if canonical(task['output']) in sources:
            raise ValueError('A correction destination overlaps an original file: ' + task['output'])
    copied = {}
    for entry in entries:
        for source in entry.get('other_files', []):
            key = canonical(Path(entry['out_dir']) / Path(source).name)
            if canonical(source) == key:
                continue
            if key in outputs or key in sources:
                raise ValueError('Companion destination overlaps an image or original: ' + key)
            if key in copied and copied[key] != canonical(source):
                raise ValueError('Companion destination has conflicting sources: ' + key)
            copied[key] = canonical(source)
    return tasks


def write_selection_report(root, run_id, report):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / ('selection_' + run_id + '.json'), report)
    with open(root / ('selection_' + run_id + '.csv'), 'x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case_id', 'status', 'details'])
        writer.writeheader()
        writer.writerows(report['records'])


def print_selection(report, tasks, entries, dry_run):
    print(f"Available folders: {report['considered_folders']} | Selected: {report['selected_folders']} | "
          f"Eligible: {report['eligible_folders']} | Skipped: {report['skipped_folders']}")
    print('Matched images by type: ' + json.dumps(report['matched_images'], sort_keys=True))
    print('Queued images by type: ' + json.dumps(report['eligible_images'], sort_keys=True))
    for record in report['records']:
        if record['status'] != 'eligible':
            print(f"[{record['status']}] {record['case_id']}: {record['details']}")
    if dry_run:
        for task in tasks:
            print(f"[plan] {task['input']} -> {task['output']} | mask={task['mask'] or 'Otsu'}")
        for entry in entries:
            for source in entry.get('other_files', []):
                target = Path(entry['out_dir']) / Path(source).name
                if canonical(source) != canonical(target):
                    print(f'[copy] {source} -> {target}')
        print('Dry run only: no N4 processing or file creation.')


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('Must be at least 1')
    return value


def visible_cpu_count():
    """Best-effort process-visible logical CPUs; not a scheduler reservation."""
    process_count = getattr(os, 'process_cpu_count', None)
    if process_count is not None:
        value = process_count()
        if value is not None and value > 0:
            return value
    affinity = getattr(os, 'sched_getaffinity', None)
    if affinity is not None:
        try:
            value = len(affinity(0))
            if value > 0:
                return value
        except (OSError, NotImplementedError):
            pass
    return os.cpu_count()


def print_cpu_budget(num_workers, itk_threads_per_worker):
    visible = visible_cpu_count()
    budget = num_workers * itk_threads_per_worker
    print(f'Workers: {num_workers} | ITK threads/worker: {itk_threads_per_worker} | '
          f'Requested computation-thread budget: {budget} | '
          f'Process-visible logical CPUs: {visible if visible is not None else "unknown"}')
    print('Thread settings are limits; simultaneous CPU utilization is not guaranteed.')
    if visible is not None and budget > visible:
        print(f'[Warning] Requested computation-thread budget {budget} exceeds '
              f'{visible} process-visible CPUs; continuing without reducing your settings.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', help='YAML configuration file')
    parser.add_argument('--dry-run', action='store_true', help='Preview selection/destinations without creating files')
    parser.add_argument('-d', '--dataset', choices=['1', '2', 'custom'])
    parser.add_argument('-i', '--input_root')
    parser.add_argument('-o', '--output_root')
    parser.add_argument('-s', '--subset')
    parser.add_argument('-w', '--num_workers', type=positive_int)
    parser.add_argument('--itk_threads_per_worker', type=positive_int,
                        help='ITK computation-thread limit per scan worker (default: 1)')
    parser.add_argument('--max_retries', type=int)
    parser.add_argument('--retry_delay', type=float)
    parser.add_argument('--shrink_factor', type=positive_int)
    parser.add_argument('--num_iters', type=positive_int)
    parser.add_argument('--num_fitting_levels', type=positive_int)
    parser.add_argument('--convergence_thresh', type=float)
    parser.add_argument('--sequences', nargs='+', help='Exact suffixes after final hyphen, e.g. t1c t1n t2f t2w')
    args = parser.parse_args(argv)
    try:
        cfg = {}
        if args.config:
            if args.dataset is not None or args.sequences is not None:
                raise ValueError('--config cannot be combined with --dataset or --sequences')
            cfg = load_config(args.config)
            input_root = args.input_root if args.input_root is not None else cfg.get('input_dir')
            output_root = args.output_root if args.output_root is not None else cfg.get('output_dir')
        elif args.input_root is not None or args.dataset == 'custom':
            input_root, output_root = args.input_root, args.output_root
        else:
            preset = DATASETS_CONFIG[args.dataset or '1']
            input_root = preset['input_root']
            output_root = args.output_root if args.output_root is not None else preset['output_root']
        input_root = resolve_path(input_root, Path.cwd(), 'input_dir')
        output_root = input_root if output_root is None else resolve_path(output_root, Path.cwd(), 'output_dir')
        if (canonical(input_root) != canonical(output_root)
                and (Path(input_root) in Path(output_root).parents or Path(output_root) in Path(input_root).parents)):
            raise ValueError('Input/output roots must be equal for in-place output, or separate non-nested directories')
        defaults = dict(num_workers=min(4, os.cpu_count() or 1), itk_threads_per_worker=1,
                        max_retries=3, retry_delay=2.0)
        settings = {key: getattr(args, key) if getattr(args, key) is not None else cfg.get(key, default)
                    for key, default in defaults.items()}
        params = {key: getattr(args, key) if getattr(args, key) is not None else cfg.get('n4_params', {}).get(key, default)
                  for key, default in DEFAULT_PARAMETERS.items()}
        for key in ('num_workers', 'itk_threads_per_worker', 'max_retries'):
            value = settings[key]
            if type(value) is not int or value < (0 if key == 'max_retries' else 1):
                raise ValueError(key + ' must be an integer ' + ('>= 0' if key == 'max_retries' else '>= 1'))
        for key in ('shrink_factor', 'num_fitting_levels', 'num_iters'):
            if type(params[key]) is not int or params[key] < 1:
                raise ValueError(key + ' must be an integer >= 1')
        for key, value in [('retry_delay', settings['retry_delay']), ('convergence_thresh', params['convergence_thresh'])]:
            if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
                raise ValueError(key + ' must be a finite nonnegative number')
        if os.name == 'nt' and settings['num_workers'] > 61:
            raise ValueError('Windows ProcessPoolExecutor supports at most 61 workers')
        if args.config:
            entries, report = select_configured_cases(input_root, output_root, cfg, args.subset)
        else:
            entries = discover_patient_files(input_root, output_root, args.subset, args.sequences)
            count = sum(len(e['structural_scans']) for e in entries)
            report = dict(considered_folders=len([p for p in Path(input_root).iterdir() if p.is_dir() and not p.name.startswith('.')]),
                          selected_folders=len(entries), eligible_folders=len(entries), skipped_folders=0,
                          matched_images={'legacy_selection': count}, eligible_images={'legacy_selection': count},
                          records=[dict(case_id=e['id'], status='eligible', details='') for e in entries])
        tasks = validate_destinations(entries)
        print(f'Input: {input_root}\nOutput: {output_root}')
        print_cpu_budget(settings['num_workers'], settings['itk_threads_per_worker'])
        print_selection(report, tasks, entries, args.dry_run)
        if args.dry_run:
            return 0 if entries else 1
        if not entries:
            run_id = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
            write_selection_report(Path(output_root) / '.n4_logs', run_id, report)
            raise ValueError('No eligible cases; see selection report in .n4_logs')
        success = logging_loop('Step3_BiasCorrection_MultiWorker', input_root, output_root,
                               entries, params, settings['num_workers'], settings['max_retries'],
                               settings['retry_delay'], selection_report=report,
                               itk_threads_per_worker=settings['itk_threads_per_worker'])
        return 0 if success else 1
    except (ValueError, OSError, RuntimeError, ScanBusy) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Interrupted. Rerun to validate and resume completed scans.', file=sys.stderr)
        sys.exit(130)
