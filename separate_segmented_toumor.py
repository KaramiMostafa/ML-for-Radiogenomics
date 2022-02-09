import os
import numpy as np
import SimpleITK as sitk


""" Separating the segmented images of task 1 into four parts:
    1. image with the whole tumor (labes : 1+2+4)
    2. image with necrotic (NCR) parts of the tumor (labes : 1)
    3. image with peritumoral edematous/invaded tissue (ED) (labes : 2)
    4. image with enhancing tumor (ET) (labes : 4)
    5. image with tumor core (COR) (labels : 1+4)
"""

#%% list of patients and input with output directory
# list of patient's ID of task 1
patientID = os.listdir(
    r"D:\ICT\Thesis\Data\Task1\RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021"
)

# path to task 1 data folder
train_task1 = (
    r"D:\ICT\Thesis\Data\Task1\RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021"
)

out_path = (
    r"D:\ICT\Thesis\Data\Task1\RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021"
)


#%% loop through the input path folder to separate different area
#   and relable it to 1 for feature extraction

for patient in patientID:

    # reading the original image
    img = sitk.ReadImage(
        os.path.join(
            train_task1,
            patient + "/" + patient + "_seg.nii.gz",
        )
    )

    # getting the image data
    img_data = sitk.GetArrayFromImage(img)

    # relabel the whole tumor to 1
    img_whole_data = np.where((img_data != 0), 1, img_data)

    # keeping the nerotic part of the tumor and removing the rest
    img_NCR_data = np.where((img_data != 1), 0, img_data)

    # extracting the edema (label 2) and relabel it to 1
    img_ED_data = np.where(
        (np.where((img_data != 2), 0, img_data) == 2),
        1,
        np.where((img_data != 2), 0, img_data),
    )
    
    # extracting the enhacing part of the tumor (label 4) and relabel it to 1
    img_ET_data = np.where(
        (np.where((img_data != 4), 0, img_data) == 4),
        1,
        np.where((img_data != 4), 0, img_data),
    )

    # extracting the core of the tumor (label 1 & 4) and relabel it to 1
    img_COR_data = (
        np.where((img_data == 2), 0, img_data)
        & np.where((img_data == 1), 1, img_data)
        & np.where((img_data == 4), 1, img_data)
    )
    
    # getting the metadata of the original image and assign it to 
    # new segmented area as NIFTI file
    img_whole = sitk.GetImageFromArray(img_whole_data)
    img_whole.CopyInformation(img)
    img_NCR = sitk.GetImageFromArray(img_NCR_data)
    img_NCR.CopyInformation(img)
    img_ED = sitk.GetImageFromArray(img_ED_data)
    img_ED.CopyInformation(img)
    img_ET = sitk.GetImageFromArray(img_ET_data)
    img_ET.CopyInformation(img)
    img_COR = sitk.GetImageFromArray(img_COR_data)
    img_COR.CopyInformation(img)


    # saving all the nifti files in output path
    sitk.WriteImage(
        img_whole, os.path.join(out_path, patient + "/" + patient + "_seg_whole.nii.gz")
    )

    sitk.WriteImage(
        img_NCR, os.path.join(out_path, patient + "/" + patient + "_seg_NCR.nii.gz")
    )

    sitk.WriteImage(
        img_ED, os.path.join(out_path, patient + "/" + patient + "_seg_ED.nii.gz")
    )

    sitk.WriteImage(
        img_ET, os.path.join(out_path, patient + "/" + patient + "_seg_ET.nii.gz")
    )

    sitk.WriteImage(
        img_COR, os.path.join(out_path, patient + "/" + patient + "_seg_COR.nii.gz")
    )
