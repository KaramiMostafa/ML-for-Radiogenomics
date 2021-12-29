import os
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib

''' getting the patient's ID list names and the MGMT status
    creating a path for each patient and each sequence type
    saving images in jpg format with printing usefull info.:
             (ID, MGMT status, sequence type)
'''
#%% Getting the train data labels 
train_df = pd.read_csv(
    "D:/ICT/Thesis/Data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
)
print(train_df)


#%% path of NIFTI files and output path for jpg images
patientID = os.listdir(
    "D:/ICT/Thesis/Data/rsna-miccai-brain-tumor-radiogenomic-classification/train_nifti/"
)
visualization_path = "D:/ICT/Thesis/Data/rsna-miccai-brain-tumor-radiogenomic-classification/visualization_images/"
train_path_nifti = "D:/ICT/Thesis/Data/rsna-miccai-brain-tumor-radiogenomic-classification/train_nifti/"
SQtypes = ["FLAIR", "T1w", "T1wCE", "T2w"]  # Sequence tpyes


#%% Which type of sequence to visualize + save it in different folder
for patient in patientID:
    for types in SQtypes:
        img = nib.load(
            os.path.join(
                train_path_nifti, patient + "/" + types + "/" + patient + types + ".nii"
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
        axes[0].imshow(slice_0.T, cmap="gray", origin="lower", aspect="auto")
        axes[0].set_title("Sagittal", fontsize=30, fontweight="bold")
        axes[1].imshow(slice_1.T, cmap="gray", origin="lower", aspect="auto")
        axes[1].set_title("Coronal", fontsize=30, fontweight="bold")
        axes[2].imshow(slice_2.T, cmap="gray", origin="lower", aspect="auto")
        axes[2].set_title("Axial", fontsize=30, fontweight="bold")
        plt.suptitle(
            f"Axial, Sagittal and Coronal view of the patient's brain \n Patient ID : {patient}    MGMT_value : {int(train_df.loc[train_df['BraTS21ID'] == int(patient)]['MGMT_value'])}    Type: {types} ",
            fontsize=38,
            fontweight="bold",
        )
        os.makedirs(visualization_path + patient + "/" + types + "/", exist_ok=True)
        fig.savefig( # saving the jpg format with info.
            visualization_path + patient + "/" + types + "/" + patient + types + ".jpg",
            dpi=300,
        )
