import matplotlib.pyplot as plt
import os
import dicom2nifti
import pandas as pd
import nibabel as nib


#%% Getting the train data labels
train_df = pd.read_csv(
    "D:/ICT/Thesis/Data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
)
print(train_df)

# Asking user for specific patient ID (based on train_df)
patientID = input("Enter patient's ID to visualize: ")
train_path = (
    "D:/ICT/Thesis/Data/rsna-miccai-brain-tumor-radiogenomic-classification/train/"
)
train_path_nifti = "D:/ICT/Thesis/Data/rsna-miccai-brain-tumor-radiogenomic-classification/train_nifti/"
SQtypes = ["FLAIR", "T1w", "T1wCE", "T2w"]  # Sequence tpyes


#%% Conversion from DICOM to NIFTI
image = []
try:
    for types in SQtypes:
        os.makedirs(train_path_nifti + patientID + "/" + types + "/", exist_ok=True)
        dicom2nifti.dicom_series_to_nifti(
            train_path + patientID + "/" + types,
            os.path.join(
                train_path_nifti, patientID + "/" + types + "/" + patientID + ".nii"
            ),
        )
        image.append(
            os.path.join(
                train_path_nifti, patientID + "/" + types + "/" + patientID + ".nii"
            )
        )
except:
    print("Incorrecet patient's ID.")

#%% Which type of sequence to visualize + showing t
try:
    tp = int(
        input(
            "Select the sequence type: \n 1. FLAIR \n 2. T1w \n 3. T1wCE \n 4. T2w \n"
        )
    )

    if tp == 1:
        img = nib.load(
            os.path.join(
                train_path_nifti,
                patientID + "/" + SQtypes[0] + "/" + patientID + ".nii",
            )
        )
        type2print = SQtypes[0]
    elif tp == 2:
        img = nib.load(
            os.path.join(
                train_path_nifti,
                patientID + "/" + SQtypes[1] + "/" + patientID + ".nii",
            )
        )
        type2print = SQtypes[1]
    elif tp == 3:
        img = nib.load(
            os.path.join(
                train_path_nifti,
                patientID + "/" + SQtypes[2] + "/" + patientID + ".nii",
            )
        )
        type2print = SQtypes[2]
    elif tp == 4:
        img = nib.load(
            os.path.join(
                train_path_nifti,
                patientID + "/" + SQtypes[3] + "/" + patientID + ".nii",
            )
        )
        type2print = SQtypes[3]
    else:
        pass

    img_data = img.get_fdata()

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
        f"Axial, Sagittal and Coronal view of the patient's brain \n Patient ID : {patientID}    MGMT_value : {train_df.loc[int(patientID)]['MGMT_value']}    Type: {type2print} ",
        fontsize=38,
        fontweight="bold",
    )

except:
    print("Incorrecet sequence type.")
