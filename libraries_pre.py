"""
libraries : data pre-processing
"""
import os
import dicom2nifti
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

""" train_df : patients lables 
    patientID : list of patiend's ID in input folder
    train_path : DICOM format path
    train_path_nifti : NIFTI format path --> output for conversion function
    visualization_path : JPG format path --> output for visualization function
    sequence_types : sequence type of mMRI pictures as a list of string
"""
class PreProcess:
    
    def __init__(
        self,
        train_df,
        patientID,
        train_path,
        train_path_nifti,
        visualization_path,
        sequence_types,
    ):

        self.train_df = pd.read_csv(
            "D:/ICT/Thesis/Data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
        )
        self.patientID = os.listdir(
            "D:/ICT/Thesis/Data/rsna-miccai-brain-tumor-radiogenomic-classification/train/"
        )
        # self.train_path = (
        #     "D:/ICT/Thesis/Data/rsna-miccai-brain-tumor-radiogenomic-classification/train/"
        # )
        # self.visualization_path = "D:/ICT/Thesis/Data/rsna-miccai-brain-tumor-radiogenomic-classification/visualization_images/"
        # self.train_path_nifti = "D:/ICT/Thesis/Data/rsna-miccai-brain-tumor-radiogenomic-classification/train_nifti/"
        self.sequence_types = ["FLAIR", "T1w", "T1wCE", "T2w"]
        
    """iterating over patient's ID and the sequence type (inner iteration)
    to convert each picture into NIFTI format and save it in "converted_path"
    """
    def conversion(self, original_path, converted_path):

        for patient in self.patientID:
            try:
                for types in self.sequence_types:
                    os.makedirs(
                        converted_path + patient + "/" + types + "/", exist_ok=True
                    )
                    dicom2nifti.dicom_series_to_nifti(
                        original_path + patient + "/" + types,
                        os.path.join(
                            converted_path,
                            patient + "/" + types + "/" + patient + types,
                        ),
                    )
            except:
                continue

    """iterating over patient's ID and the sequence type (inner iteration)
    to convert each picture into JPG format and save it in "jpg_path" for visualization
    containing : MGMT_value status for each one
    """
    def visualization(self, nifti_path, jpg_path):

        for patient in self.patientID:
            try:
                for types in self.sequence_types:
                    img = nib.load(
                        os.path.join(
                            nifti_path,
                            patient + "/" + types + "/" + patient + types + ".nii",
                        )
                    )
                    img_data = img.get_fdata()

                    # extracting different slice of the brain for three dircetion views
                    slice_0 = img_data[img.shape[0] // 2, :, :]
                    slice_1 = img_data[:, img.shape[1] // 2, :]
                    slice_2 = img_data[:, :, img.shape[2] // 2]

                    # plotting 3 views
                    fig, axes = plt.subplots(1, 3)
                    fig.set_figheight(17)
                    fig.set_figwidth(40)
                    axes[0].imshow(
                        slice_0.T, cmap="gray", origin="lower", aspect="auto"
                    )
                    axes[0].set_title("Sagittal", fontsize=30, fontweight="bold")
                    axes[1].imshow(
                        slice_1.T, cmap="gray", origin="lower", aspect="auto"
                    )
                    axes[1].set_title("Coronal", fontsize=30, fontweight="bold")
                    axes[2].imshow(
                        slice_2.T, cmap="gray", origin="lower", aspect="auto"
                    )
                    axes[2].set_title("Axial", fontsize=30, fontweight="bold")
                    plt.suptitle(
                        f"Axial, Sagittal and Coronal view of the patient's brain \n Patient ID : {patient}    MGMT_value : {int(self.train_df.loc[self.train_df['BraTS21ID'] == int(patient)]['MGMT_value'])}    Type: {types} ",
                        fontsize=38,
                        fontweight="bold",
                    )
                    os.makedirs(jpg_path + patient + "/" + types + "/", exist_ok=True)
                    fig.savefig(  # saving the jpg format with info.
                        jpg_path
                        + patient
                        + "/"
                        + types
                        + "/"
                        + patient
                        + types
                        + ".jpg",
                        dpi=300,
                    )
            except:
                continue
    
    """
    This function finds the best postion view in each direction of sagittal,
    coronal and axial. It is done by counting all the non-zero cell and storing the 
    maximum value for each direction.
    
    Args:
    img : NIFTI image as an input
    
    return --> best_postions as a list containg the best postion view for visualization
    """
    def best_view(self,img):
        
        img_data = img.get_fdata()
        count_sag, count_axi, count_cor = [],[],[]
        
        for i in range(0,img.shape[0]):
            count_sag.append(np.count_nonzero(img_data[i, : , :]))
            
        for j in range(0,img.shape[1]):
            count_cor.append(np.count_nonzero(img_data[:, j , :]))
    
        for k in range(0,img.shape[2]):
            count_axi.append(np.count_nonzero(img_data[:, : , k]))
        
        position_cor = np.argmax(count_cor) 
        position_sag = np.argmax(count_sag)
        position_axi = np.argmax(count_axi)
            
        return [position_sag, position_cor, position_axi]
    
    
    ''' <<image_info>> function extracts the volume and size of images + single 
        vocxel in each one and store it in a csv file.
    '''
    def image_info(self):
        
        df = pd.DataFrame(
            columns=["image", "voxel volume", "voxel size", "image volume", "image size"]
        )

        for patient in self.patientID:
            try:
                for types in self.sqtypes_task2:
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
                           self.train_path_nifti,
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