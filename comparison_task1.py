import matplotlib.pyplot as plt
import os
import pandas as pd
import nibabel as nib


#%% Getting the train data labels
train_df = pd.read_csv(
    "D:/ICT/Thesis/Data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
)
print(train_df)


#%% path of DICOM files and output path for NIFTI
patientID = os.listdir(
    "D:/ICT/Thesis/Data/rsna-miccai-brain-tumor-radiogenomic-classification/incase/"
)
train_path_nifti = "D:/ICT/Thesis/Data/rsna-miccai-brain-tumor-radiogenomic-classification/incase/"
train_task1 = (
    "D:/ICT/Thesis/Data/Task1/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/"
)
comparison_path = "D:/ICT/Thesis/Result/comparison_with_task1/"

sqtypes_task1 = ["flair", "t1", "t1ce", "t2"]
sqtypes_task2 = ["FLAIR", "T1w", "T1wCE", "T2w"]

for patient in patientID:
    # try:
        for type_task1, type_task2 in zip(sqtypes_task1, sqtypes_task2):

            img_task1 = nib.load(
                os.path.join(
                    train_task1,
                    "BraTS2021_"
                    + patient
                    + "/"
                    + "BraTS2021_"
                    + patient
                    + "_"
                    + type_task1
                    + ".nii.gz",
                )
            )

            img_data_task1 = img_task1.get_fdata()

            img_task2 = nib.load(
                os.path.join(
                    train_path_nifti,
                    patient + "/" + type_task2 + "/" + patient + type_task2 + ".nii",
                )
            )

            img_data_task2 = img_task2.get_fdata()

            # extracting different slice of the brain for three dircetion views
            slice1_task1 = img_data_task1[img_task1.shape[0] // 2, :, :]
            slice2_task1 = img_data_task1[:, img_task1.shape[1] // 2, :]
            slice3_task1 = img_data_task1[:, :, img_task1.shape[2] // 2]

            slice1_task2 = img_data_task2[img_task2.shape[0] // 2, :, :]
            slice2_task2 = img_data_task2[:, img_task2.shape[1] // 2, :]
            slice3_task2 = img_data_task2[:, :, img_task2.shape[2] // 2]

            # plotting 3 views
            fig, axes = plt.subplots(2, 3)
            fig.set_figheight(17)
            fig.set_figwidth(40)
            axes[0][0].imshow(
                slice1_task2.T, cmap="gray", origin="lower", aspect="auto"
            )
            axes[0][0].set_title("Sagittal", fontsize=30, fontweight="bold")
            axes[0][1].imshow(
                slice2_task2.T, cmap="gray", origin="lower", aspect="auto"
            )
            axes[0][1].set_title("Coronal", fontsize=30, fontweight="bold")
            axes[0][2].imshow(
                slice3_task2.T, cmap="gray", origin="lower", aspect="auto"
            )
            axes[0][2].set_title("Axial", fontsize=30, fontweight="bold")

            axes[1][0].imshow(
                slice1_task1.T, cmap="gray", origin="lower", aspect="auto"
            )
            axes[1][0].set_title("Sagittal", fontsize=30, fontweight="bold")
            axes[1][1].imshow(
                slice2_task1.T, cmap="gray", origin="lower", aspect="auto"
            )
            axes[1][1].set_title("Coronal", fontsize=30, fontweight="bold")
            axes[1][2].imshow(
                slice3_task1.T, cmap="gray", origin="lower", aspect="auto"
            )
            axes[1][2].set_title("Axial", fontsize=30, fontweight="bold")

            plt.suptitle(
                f"Axial, Sagittal and Coronal view of the patient's brain \n Patient ID : {patient}    MGMT_value : {int(train_df.loc[train_df['BraTS21ID'] == int(patient)]['MGMT_value'])}    Type: {type_task2} ",
                fontsize=38,
                fontweight="bold",
            )
            os.makedirs(
                comparison_path + patient + "/" + type_task2 + "/", exist_ok=True
            )
            fig.savefig(  # saving the jpg format with info.
                comparison_path
                + patient
                + "/"
                + type_task2
                + "/"
                + patient
                + type_task2
                + ".jpg",
                dpi=300,
            )
    # except:
    #     continue
