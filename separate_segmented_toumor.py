import os
import nibabel as nib
import numpy as np
    


''' Separating the segmented images of task 1 into four parts:
    1. image with the whole tumor (labes : 1+2+4)
    2. image with necrotic (NCR) parts of the tumor (labes : 1)
    3. image with peritumoral edematous/invaded tissue (ED) (labes : 2)
    4. image with enhancing tumor (ET) (labes : 4)
'''

# list of patient's ID of task 1
patientID = os.listdir(
    "D:/ICT/Thesis/Data/Task1/incase/"
)

# path to task 1 data folder 
train_task1 = (
    "D:/ICT/Thesis/Data/Task1/incase/"
)



for patient in patientID:
    
    img = nib.load(
        os.path.join(
            train_task1,
            patient
            + "/"
            + patient
            + "_seg.nii.gz",
        )
    )
    
    img_data = img.get_fdata()
    
    img_NCR_data = np.where(
        (img_data != 1 ), 0, img_data
    )
    
    img_ED_data = np.where(
        (img_data != 2 ), 0, img_data
    )
    
    img_ET_data = np.where(
        (img_data != 4 ), 0, img_data
    )
    
    
    img_NCR = nib.Nifti1Image(img_NCR_data, img.affine, img.header)
    img_ED = nib.Nifti1Image(img_ED_data, img.affine, img.header)
    img_ET = nib.Nifti1Image(img_ET_data, img.affine, img.header)
    
    nib.save(img_NCR, os.path.join(
            train_task1,
            patient
            + "/"
            + patient
            + "_seg_NCR.nii.gz",
        )
    )
    
    nib.save(img_ED, os.path.join(
            train_task1,
            patient
            + "/"
            + patient
            + "_seg_ED.nii.gz",
        )
    )
    
    nib.save(img_ET, os.path.join(
            train_task1,
            patient
            + "/"
            + patient
            + "_seg_ET.nii.gz",
        )
    )
    
    

    
    