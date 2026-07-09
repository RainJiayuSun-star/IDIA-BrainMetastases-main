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
#     do labels combining, and save them as [patient_id]_combined.seg.nii
# if the patient only contains the seg.nii segmentation
#     save it as [patient_id]_combined.seg.nii
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
            "combined_status": "Not processed",
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

        combined_out_name = f"{patientid}_combined.seg.nii.gz"

        # check if it contains the brats segmentation file        
        # if it does contain, then combine the labels, save the combined seg file
        if bratsSeg_file in patientfiles_names:
            bratsSeg_path = os.path.join(patientDirPath, bratsSeg_file)
            bratsSeg_image = sitk.ReadImage(bratsSeg_path)               # Correct path
            bratsSeg_array = sitk.GetArrayFromImage(bratsSeg_image)      # Correct image object

            # combine necrosis (1) and enhancing tumor (3/4) to Tumor Core (Total Tumor Volume = 1)
            # and edema (2) to Edema (2)
            combined_array = np.zeros_like(bratsSeg_array, dtype=np.uint8)
            combined_array[np.isin(bratsSeg_array, [1, 3, 4])] = 1
            combined_array[bratsSeg_array == 2] = 2

            # save Combined Segmentations preserving spatial metadata
            combined_img = sitk.GetImageFromArray(combined_array)
            combined_img.CopyInformation(bratsSeg_image)
            sitk.WriteImage(combined_img, os.path.join(patientDirPath, combined_out_name))
            
            record["combined_status"] = "Combined & Saved"
            record["status"] = "Processed (BraTS)"

        # if it does not contain BraTS-seg but contains standard seg, save as tumorcore
        elif seg_file in patientfiles_names:
            seg_img_path = os.path.join(patientDirPath, seg_file)
            seg_image = sitk.ReadImage(seg_img_path)
            
            # Save standard seg as tumorcore directly (saving as .nii.gz)
            tumorcore_out_name = f"{patientid}_tumorcore.seg.nii.gz"
            sitk.WriteImage(seg_image, os.path.join(patientDirPath, tumorcore_out_name))
            
            record["combined_status"] = "Saved as tumorcore"
            record["status"] = "Processed (Standard Seg)"

        # if it contains neither
        else:
            record["combined_status"] = "Missing"
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