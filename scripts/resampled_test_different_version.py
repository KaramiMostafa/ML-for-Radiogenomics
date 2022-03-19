import matplotlib.pyplot as plt
import nibabel as nib
import nibabel.processing
import SimpleITK as sitk
from resample import *

''' Comparison between resampled versions of Task 2 and Task 1 using "resample" 
    file and nibabel library. 
'''


#%% path of NIFTI images of Task 2 and Task 1 and output path JPG as a comparison
# reaeding task 2 image with different libraries with their data 
img_task1 = nib.load(r'D:\ICT\Thesis\Data\Task1\RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021\BraTS2021_00558\BraTS2021_00558_t1ce.nii.gz')
img_task2_sitk = sitk.ReadImage(r'D:\ICT\Thesis\Data\rsna-miccai-brain-tumor-radiogenomic-classification\train_nifti\00558\T1wCE\00558T1wCE.nii')
img_task2_nib = nib.load(r'D:\ICT\Thesis\Data\rsna-miccai-brain-tumor-radiogenomic-classification\train_nifti\00558\T1wCE\00558T1wCE.nii')

# output directory to save the comparison result 
out_path = r'D:\ICT\Thesis\Github\repo\Resample\00558T1wCE.jpg'


# getting resampled image with data - SimpleITK version
img_task2_sitk = resample_image(img_task2_sitk, out_spacing= (1, 1, 1),
                           out_size= None,
                           interpolator= sitk.sitkBSpline)
img_data_task2_sitk = sitk.GetArrayFromImage(img_task2_sitk)


# getting resampled image with data - nibabel version
img_task2_nib = nibabel.processing.conform(img_task2_nib, out_shape=(240, 240, 155)
                                             , voxel_size=(1.0, 1.0, 1.0),
                                             orientation='LPS')
img_data_task2_nib = img_task2_nib.get_fdata()

# data image - task 1
img_data_task1 = img_task1.get_fdata()


#%% extracting different slice of the brain for three view dircetion
# task 1
slice1_task1 = img_data_task1[img_task1.shape[0] // 2, :, :]
slice2_task1 = img_data_task1[:, img_task1.shape[1] // 2, :]
slice3_task1 = img_data_task1[:, :, img_task1.shape[2] // 2]

# resampled task 2 - SimpleITK version
slice1_task2_sitk = img_data_task2_sitk[img_data_task2_sitk.shape[0] // 2, :, :]
slice2_task2_sitk = img_data_task2_sitk[:, img_data_task2_sitk.shape[1] // 2, :]
slice3_task2_sitk = img_data_task2_sitk[:, :, img_data_task2_sitk.shape[2] // 2]

# resampled task 2 - nibabel version
slice1_task2_nib = img_data_task2_nib[img_data_task2_nib.shape[0] // 2, :, :]
slice2_task2_nib = img_data_task2_nib[:, img_data_task2_nib.shape[1] // 2, :]
slice3_task2_nib = img_data_task2_nib[:, :, img_data_task2_nib.shape[2] // 2]

# plotting 3 views
fig, axes = plt.subplots(3, 3)
fig.set_figheight(40)
fig.set_figwidth(60)

# Task 2 : with SimpleITK, imported from resample file
axes[0][0].imshow(
    slice1_task2_sitk.T, origin="lower", aspect="auto"
)
axes[0][0].set_title("Sagittal", fontsize=40, fontweight="bold")

axes[0][1].imshow(
    slice2_task2_sitk.T, origin="lower", aspect="auto"
)
axes[0][1].set_title("Coronal", fontsize=40, fontweight="bold")

axes[0][2].imshow(
    slice3_task2_sitk.T, origin="lower", aspect="auto"
)
axes[0][2].set_title("Axial", fontsize=40, fontweight="bold")

# Task 2 : with nibabel, using nibabel.processing.conform
# https://nipy.org/nibabel/reference/nibabel.processing.html

axes[1][0].imshow(
    slice1_task2_nib.T, origin="lower", aspect="auto"
)
axes[1][0].set_title("Sagittal", fontsize=40, fontweight="bold")
axes[1][1].imshow(
    slice2_task2_nib.T, origin="lower", aspect="auto"
)
axes[1][1].set_title("Coronal", fontsize=40, fontweight="bold")
axes[1][2].imshow(
    slice3_task2_nib.T, origin="lower", aspect="auto"
)
axes[1][2].set_title("Axial", fontsize=40, fontweight="bold")


# Task 1
axes[2][0].imshow(
    slice1_task1.T, origin="lower", aspect="auto"
)
axes[2][0].set_title("Sagittal", fontsize=40, fontweight="bold")
axes[2][1].imshow(
    slice2_task1.T, origin="lower", aspect="auto"
)
axes[2][1].set_title("Coronal", fontsize=40, fontweight="bold")
axes[2][2].imshow(
    slice3_task1.T, origin="lower", aspect="auto"
)
axes[2][2].set_title("Axial", fontsize=40, fontweight="bold")

plt.suptitle(
    "Resampled version of Task 2 by SimpleITK ( first row) and by Nibabel (secobd row) with Task 1 (third row) \n Patient ID : 00558   MGMT_value : 1    Type: T1wCE ",
    fontsize=55,
    fontweight="bold",
)

fig.savefig(out_path,
    dpi=300,
)
