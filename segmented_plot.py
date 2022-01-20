import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np 
from best_tumor_view import *

''' plotting the segmented images from Task 1 with best view of the whole tumor
    and only the core with help of function best_vew(). Ressults are stored as 
    JPG file.
'''

#%% path of NIFTI files and output path for jpg images
patientID = os.listdir(
    "D:/ICT/Thesis/Data/Task1/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/"
)
visualization_path = "D:/ICT/Thesis/Result/segmented_task1/"
segmented_path = "D:/ICT/Thesis/Data/Task1/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/"


#%% different type of sequence to visualize + save it in different folder
for patient in patientID:
        # whole tumor
        img = nib.load(
            os.path.join(
                segmented_path,
                patient
                + "/"
                + patient
                + "_seg.nii.gz",
            )
        )
        
        img_core = img
        
        
        img_data = img.get_fdata()
        best_postion = best_view(img)
        
        # only core of the tumor
        img_core_data = img_core.get_fdata()
        img_core_data = np.where(
            (img_data == 4), 0, img_core_data
        )

        # assigning best position in each direction for visializing tumore core

        best_postions_core = best_view(img_core)
                
                
        # extracting different slice of the brain for three dircetion views
        slice_0 = img_data[best_postion[0], :, :]
        slice_1 = img_data[:, best_postion[1], :]
        slice_2 = img_data[:, :, best_postion[2]]
        
        
        slice_0_core = img_core_data[best_postions_core[0], :, :]
        slice_1_core = img_core_data[:, best_postions_core[1], :]
        slice_2_core = img_core_data[:, :, best_postions_core[2]]
        

        # plotting 3 views
        fig, axes = plt.subplots(2, 3)
        fig.set_figheight(20)
        fig.set_figwidth(40)
        
        axes[0][0].imshow(slice_0.T, origin="lower", aspect="auto")
        axes[0][0].set_title("Sagittal", fontsize=30, fontweight="bold")
        axes[0][1].imshow(slice_1.T, origin="lower", aspect="auto")
        axes[0][1].set_title("Coronal", fontsize=30, fontweight="bold")
        axes[0][2].imshow(slice_2.T, origin="lower", aspect="auto")
        axes[0][2].set_title("Axial", fontsize=30, fontweight="bold")
        
        axes[1][0].imshow(slice_0_core.T, origin="lower", aspect="auto")
        axes[1][0].set_title("Sagittal", fontsize=30, fontweight="bold")
        axes[1][1].imshow(slice_1_core.T, origin="lower", aspect="auto")
        axes[1][1].set_title("Coronal", fontsize=30, fontweight="bold")
        axes[1][2].imshow(slice_2_core.T, origin="lower", aspect="auto")
        axes[1][2].set_title("Axial", fontsize=30, fontweight="bold")
        
        plt.suptitle(
            f"Best view of the whole tumor (first row) and only the core (second row) \n Patient ID : {patient}",
            fontsize=38,
            fontweight="bold",
        )
        # os.makedirs(visualization_path + patient + "/" + types + "/", exist_ok=True)
        fig.savefig(  # saving the jpg format with info.
            visualization_path
            + patient
            + ".jpg",
            dpi=300,
        )

