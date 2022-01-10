import nibabel as nib
import numpy as np
import pandas as pd
import os

#%% path of NIFTI
""" patieentID: list of BraTS21ID in train data (NIFTI formtat) 
    train_path_nifti: NIFTI images as a path
    sqtypes: list of sequence type 
"""
sqtypes = ["FLAIR", "T1w", "T1wCE", "T2w"]  # Sequence tpyes
train_path_nifti = "D:/ICT/Thesis/Data/rsna-miccai-brain-tumor-radiogenomic-classification/train_nifti/"
patientID = os.listdir(
    "D:/ICT/Thesis/Data/rsna-miccai-brain-tumor-radiogenomic-classification/train_nifti/"
)


#%% df --> dataframe containing each image's desired results
df = pd.DataFrame(
    columns=["image", "voxel volume", "voxel size", "image volume", "image size"]
)


#%% Getting image size and voxel size each patients
""" using try/except technique is usefull since in some cases
over sequence type iteration, there will be an error form mentioned patients
"""
for patient in patientID:
    try:
        for types in sqtypes:
            img = nib.load(
                train_path_nifti
                + patient
                + "/"
                + types
                + "/"
                + patient
                + types
                + ".nii"
            )
            voxel_size = list(img.header.get_zooms())
            voxel_volume = np.prod(img.header["pixdim"][1:4])

            image_shape = list(img.shape)
            voxel_count = np.count_nonzero(img.get_data() == 1)
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


df.to_csv("ImageInfo.csv", index=False)
