import nibabel as nib
import numpy as np
import pandas as pd
import os

''' This section extracts the volume and size of images + single vocxel in 
    each one for both Task 1 nad 2 and store it in a csv file.
'''
#%% path of NIFTI images
""" patieentID: list of BraTS21ID in train data (NIFTI formtat) 
    train_path_nifti: NIFTI images as a path
    sqtypes: list of sequence type 
"""
# sqtypes_task1 = ["flair", "t1", "t1ce", "t2"]
sqtypes_task2 = ["FLAIR", "T1w", "T1wCE", "T2w"]

# train_task1 = (
#     "D:/ICT/Thesis/Data/Task1/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/")
train_path_nifti = "D:/ICT/Thesis/Result/resampled_version_task2/"


# patientID = os.listdir(
#      "D:/ICT/Thesis/Data/Task1/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/")
patientID = os.listdir(
    "D:/ICT/Thesis/Result/resampled_version_task2/"
)




#%% df --> dataframe containing each image's desired results
df = pd.DataFrame(
    columns=["image", "voxel volume", "voxel size", "image volume", "image size"]
)

#%% Getting image size and voxel size each patients

for patient in patientID:
    try:
        for types in sqtypes_task2:
            # img = nib.load(
            #     train_path_nifti
            #     + patient
            #     + "/"
            #     + patient
            #     + '_'
            #     + types
            #     + ".nii.gz"
            # )
            img = nib.load(
               os.path.join(
                   train_path_nifti,
                   patient + "/" + types + "/" + patient + types
                   + ".nii",
               )
           )
            
            voxel_size = list(img.header.get_zooms())
            voxel_volume = np.prod(img.header["pixdim"][1:4])

            image_shape = list(img.shape)
            voxel_count = np.count_nonzero(img.get_data())
            image_volume = voxel_volume * voxel_count
            image_size = (
                image_shape[0] * voxel_size[0],
                image_shape[1] * voxel_size[1],
                image_shape[2] * voxel_size[2],
            )

            df = df.append(
                {
                    "image": patient + types,
                    "voxel volume": voxel_volume,
                    "voxel size": voxel_size,
                    "image volume": image_volume,
                    "image size": image_size,
                },
                ignore_index=True,
            )
    except:
        continue

df.to_csv("imageInfo_resampled_task2_data.csv", index=False)
# df.to_csv("imageInfo_task2_data.csv", index=False)
# df.to_csv("imageInfo_task1_data.csv", index=False)
