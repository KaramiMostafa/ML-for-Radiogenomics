"""
libraries
"""
import os
import dicom2nifti
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib


class PreProcess:
    def __init__(
        self,
        train_df,
        patientID,
        train_path,
        train_path_nifti,
        visualization_path,
        sqtypes,
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
        self.sqtypes = ["FLAIR", "T1w", "T1wCE", "T2w"]

    def conversion(self, original_path, converted_path):

        for patient in self.patientID:
            try:
                for types in self.sqtypes:
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

    def visualization(self, nifti_path, jpg_path):

        for patient in self.patientID:
            try:
                for types in self.sqtypes:
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
