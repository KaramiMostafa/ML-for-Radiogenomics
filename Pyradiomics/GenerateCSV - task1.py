import os
import csv
import fnmatch

path_result = r'D:\ICT\Thesis\Github\repo\Pyradiomics'
seg_path = r'D:\ICT\Thesis\Data\Task1\RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021'
train_task1 = (r'D:\ICT\Thesis\Data\Task1\RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021')
sequence_types = ["flair", "t1", "t1ce", "t2"]

patientID = os.listdir(
            r'D:\ICT\Thesis\Data\Task1\RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021')


with open(os.path.join(path_result,'radiomics_features_task1.csv'),'a',newline='') as csvfile:
    
#     # creating the column heads
    writer = csv.writer(csvfile)
    writer.writerow(['Image','Mask'])
    
    # filling each cell with the path of image and mask
    
    try:
        for patient in patientID:
            for types in sequence_types:
                
                
                img = os.path.join(train_task1, patient + '\\'
                                   + patient +'_'+ types + '.nii.gz')
                
                
                mask = os.path.join(train_task1, patient + '\\' + 
                                     patient + '_seg.nii.gz')
                mask_ED = os.path.join(train_task1, patient + '\\' + 
                                     patient + '_seg_ED.nii.gz')
                mask_ET = os.path.join(train_task1, patient + '\\' + 
                                     patient + '_seg_ET.nii.gz')
                mask_NCR = os.path.join(train_task1, patient + '\\' + 
                                     patient + '_seg_NCR.nii.gz')
                
                writer.writerow([img] + [mask])
                writer.writerow([img] + [mask_ED])
                writer.writerow([img] + [mask_ET])
                writer.writerow([img] + [mask_NCR])
    except:
        pass
