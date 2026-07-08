"""
This file contains python code that uses SimpleITK, and numpy to combine and save segmentation labels.

"""

import numpy as np  # numpy
import pandas as pd # pandas for creating dataframes
import SimpleITK as sitk # SimpleITK
import os

# data paths and patients
data_path = "/Users/rains/rainworkspace/2-IDIALab/Datasets/UCSF_BrainMetastases_v1.3/UCSF_BrainMetastases_TRAIN/"

patientIDs = sorted(os.listdir(data_path)) # patientID list
print(patientIDs[:5]) # preview the patients ids

# iteration through each patient
# if the patient contains the brats segmentation
#     do labels combining, and save them as [patient_id]_tumorcore.seg.nii and [patient_id]_edema.seg.nii
# if the patient only contains the seg.nii segmentation
#     save it as [patient_id]_tumorcore.seg.nii
# if the patient contains neither
#     show error and catch

num_patients = 1 # int number of patients to process
patient_dataframe = pd.DataFrame() # initialize patient_dataframe, to store status of the patient
patient_record = []

#  method, for loop through number of patients, see labels, no return
def viewPatientFiles(num_patients=None):
    global patient_dataframe, patient_record  # Access the global variables
    patient_record = []                       # Reset records for this run

    for i in range(0, num_patients):
        patientid = patientIDs[i] # string, accessing the ith element of patientID array
        patientDirPath = os.path.join(data_path, patientid)

        # initialize status tracking record
        record = {
            "patient_id": patientid,
            "tumorcore_status": "Not processed",
            "edema_status": "Not processed",
            "status": "Success"
        }

        # safely handle missing patient directory
        if not os.path.exists(patientDirPath):
            record["status"] = "Error: Directory not found"
            patient_record.append(record)
            continue

        patientfiles_names = os.listdir(patientDirPath) # all the files in each patient's directory

        bratsSeg_file = f"{patientid}_BraTS-seg.nii.gz"
        seg_file = f"{patientid}_seg.nii.gz" 

        tumorecore_out_name = f"{patientid}_tumorcore.seg.nii"
        edema_out_name = f"{patientid}_edema.seg.nii"

        # check if it contains the brats segmentation file        
        # if it does contain, then combine the label, save the seg files
        if bratsSeg_file in patientfiles_names:
            bratsSeg_path = os.path.join(patientDirPath, bratsSeg_file)
            bratsSeg_image = sitk.ReadImage(bratsSeg_path)               # Correct path
            bratsSeg_array = sitk.GetArrayFromImage(bratsSeg_image)      # Correct image object

            # combine necrosis (1) and enhancing tumor (3/4) to Tumor Core (Total Tumor Volume)
            tumorcore_array = np.zeros_like(bratsSeg_array, dtype=np.uint8)
            tumorcore_array[np.isin(bratsSeg_array, [1, 3, 4])] = 1

            # save Tumor Core preserving spatial metadata
            tumorecore_img = sitk.GetImageFromArray(tumorcore_array)
            tumorecore_img.CopyInformation(bratsSeg_image)
            sitk.WriteImage(tumorecore_img, os.path.join(patientDirPath, tumorecore_out_name))

            # extract edema (2) as binary mask
            edema_array = np.zeros_like(bratsSeg_array, dtype=np.uint8)
            edema_array[bratsSeg_array == 2] = 1

            # save Edema preserving spatial metadata
            edema_img = sitk.GetImageFromArray(edema_array)
            edema_img.CopyInformation(bratsSeg_image)                    # Correct image object
            sitk.WriteImage(edema_img, os.path.join(patientDirPath, edema_out_name))
            
            record["tumorcore_status"] = "Combined & Saved"
            record["edema_status"] = "Extracted & Saved"
            record["status"] = "Processed (BraTS)"

        # if it does not contain BraTS-seg but contains standard seg, save the seg file directly
        elif seg_file in patientfiles_names:
            seg_img_path = os.path.join(patientDirPath, seg_file)
            seg_image = sitk.ReadImage(seg_img_path)
            
            # Save standard seg as tumorcore directly
            sitk.WriteImage(seg_image, os.path.join(patientDirPath, tumorecore_out_name))
            
            record["tumorcore_status"] = "Saved directly"
            record["edema_status"] = "N/A"
            record["status"] = "Processed (Standard Seg)"

        # if it contains neither
        else:
            record["tumorcore_status"] = "Missing"
            record["edema_status"] = "Missing"
            record["status"] = "Error: No segmentation files found"
            print(f"Warning: Patient {patientid} has no segmentation files.")

        patient_record.append(record)

    # Rebind to the global patient_dataframe
    patient_dataframe = pd.DataFrame(patient_record)

def combineLabels(num_patients=None):
    if num_patients is None:
        num_patients = len(patientIDs)
        viewPatientFiles(num_patients)
        return

    viewPatientFiles(num_patients)

    
combineLabels(num_patients=5)
print("Combining Job Tracking Dataframe: patient_dataframe")
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(patient_dataframe)

print("Dataframe Saved to Current Directory")
patient_dataframe.to_csv('./LabelCombineTracking.csv', index=False)