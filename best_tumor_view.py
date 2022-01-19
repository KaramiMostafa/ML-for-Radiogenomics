import matplotlib.pyplot as plt
import os
import nibabel as nib
import numpy as np

""" This function finds the best postion view in each direction of sagittal,
    coronal and axial. It is done by counting all the non-zero cell and storing the 
    maximum value for each direction.
    
    Args:
    img : NIFTI image as an input
    
    return --> best_postions as a list containg the best postion view for visualization
"""

def best_view(img):
    
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
''' iterating through the task 1 dataset with different sequence type to save the image
    and compare to the segmented image with best tumor view.
'''
if __name__=="__main__":
    # getting the patients ID
    patientID = os.listdir(
        "D:/ICT/Thesis/Data/Task1/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/"
    )
    
    # path of NIFTI images and jpg for saving the results 
    train_path_task1 = (
        "D:/ICT/Thesis/Data/Task1/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/"
    )
    best_view_path = "D:/ICT/Thesis/Result/best_view/"
    sqtypes_task1 = ["flair", "t1", "t1ce", "t2"]
    
    for patient in patientID:  # list of patients for Task 1 dataset
        try:
            for type_task1 in sqtypes_task1:  # different sequence type
    
                img_task1 = nib.load(
                    os.path.join(
                        train_path_task1,
                        patient + "/" + patient + "_" + type_task1 + ".nii.gz",
                    )
                )
    
                # getting data of the task 1 image
                img_task1_data = img_task1.get_fdata()
    
                img_task1_segmented = nib.load(
                    os.path.join(
                        train_path_task1,
                        patient + "/" + patient + "_seg.nii.gz",
                    )
                )
    
                # getting data of the segmented image
                img_task1_segmented_data = img_task1_segmented.get_fdata()
    
                # filter the image and keep only the tumor core
                # tumor core --> labels 1 + 2 (replacing label 4 with 0)
                img_task1_segmented_data = np.where(
                    (img_task1_segmented_data == 4), 0, img_task1_segmented_data
                )
    
                # assigning best position in each direction for visializing tumore core
                best_postions = best_view(img_task1_segmented)
    
                # extracting different slice of the brain for three dircetion views
                # for both segmented and normal image of task 1 dataset
                slice1_task1 = img_task1_data[best_postions[0], :, :]
                slice2_task1 = img_task1_data[:, best_postions[1], :]
                slice3_task1 = img_task1_data[:, :, best_postions[2]]
    
                slice1_task1_segmented = img_task1_segmented_data[best_postions[0], :, :]
                slice2_task1_segmented = img_task1_segmented_data[:, best_postions[1], :]
                slice3_task1_segmented = img_task1_segmented_data[:, :, best_postions[2]]
    
                # plotting 3 views
                fig, axes = plt.subplots(2, 3)
                fig.set_figheight(20)
                fig.set_figwidth(40)
    
                # axes belongs to the task 1 data as a "first row" of visualized images
                axes[0][0].imshow(
                    slice1_task1.T, cmap="gray", origin="lower", aspect="auto"
                )
                axes[0][0].set_title("Sagittal", fontsize=30, fontweight="bold")
                axes[0][1].imshow(
                    slice2_task1.T, cmap="gray", origin="lower", aspect="auto"
                )
                axes[0][1].set_title("Coronal", fontsize=30, fontweight="bold")
                axes[0][2].imshow(
                    slice3_task1.T, cmap="gray", origin="lower", aspect="auto"
                )
                axes[0][2].set_title("Axial", fontsize=30, fontweight="bold")
    
                # axes belongs to the task 1 data (segmented) as a "second row" of visualized images
                axes[1][0].imshow(
                    slice1_task1_segmented.T, cmap="gray", origin="lower", aspect="auto"
                )
                axes[1][0].set_title("Sagittal", fontsize=30, fontweight="bold")
                axes[1][1].imshow(
                    slice2_task1_segmented.T, cmap="gray", origin="lower", aspect="auto"
                )
                axes[1][1].set_title("Coronal", fontsize=30, fontweight="bold")
                axes[1][2].imshow(
                    slice3_task1_segmented.T, cmap="gray", origin="lower", aspect="auto"
                )
                axes[1][2].set_title("Axial", fontsize=30, fontweight="bold")
    
                # saving the result
                plt.suptitle(
                    f" Best view of the tumor \n Patient ID : {patient} Type: {type_task1} ",
                    fontsize=38,
                    fontweight="bold",
                )
                os.makedirs(best_view_path, exist_ok=True)
                fig.savefig(  # saving the jpg format with info.
                    best_view_path + patient + type_task1 + "_segmentedTask1" + ".jpg",
                    dpi=300,
                )
        except:
            continue
