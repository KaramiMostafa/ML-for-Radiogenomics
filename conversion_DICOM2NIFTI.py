import os
import dicom2nifti
import pandas as pd


""" Some of the data could not be converted to NIFTI format: (should be excluded till debugging)
Patient ID | Sequence Type
  00107          T2w
  00108          T2w
  00120          T2w
  00132          T2w
  00137          T2w
  00143          T2w
  00147          T2w
  00148         T1wCE
  00524          T2w
  00834         T1wCE
  01010         T1wCE
  
error :
    raise ConversionValidationError('SLICE_INCREMENT_INCONSISTENT')
    ConversionValidationError: SLICE_INCREMENT_INCONSISTENT
"""

#%% Getting the train data labels
train_df = pd.read_csv(
    "D:/ICT/Thesis/Data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
)
print(train_df)


#%% path of DICOM files and output path for NIFTI
patientID = os.listdir(
    "D:/ICT/Thesis/Data/rsna-miccai-brain-tumor-radiogenomic-classification/train/"
)
train_path = (
    "D:/ICT/Thesis/Data/rsna-miccai-brain-tumor-radiogenomic-classification/train/"
)
train_path_nifti = "D:/ICT/Thesis/Data/rsna-miccai-brain-tumor-radiogenomic-classification/train_nifti/"
SQtypes = ["FLAIR", "T1w", "T1wCE", "T2w"]  # Sequence tpyes


#%% Conversion from DICOM to NIFTI
for patient in patientID:
    try:
        for types in SQtypes:
            os.makedirs(train_path_nifti + patient + "/" + types + "/", exist_ok=True)
            dicom2nifti.dicom_series_to_nifti(
                train_path + patient + "/" + types,
                os.path.join(
                    train_path_nifti, patient + "/" + types + "/" + patient + types
                ),
            )
    except:
        continue
