import pydicom
import pandas as pd
import os
import random


#%% extracting patient's ID in 5 digits
patients=[]
train_path = 'D:/ICT/Thesis/Data/rsna-miccai-brain-tumor-radiogenomic-classification/train'
for file in os.listdir(train_path):
    patients.append(file)

#%% tarin data labels + path to it
train_df = pd.read_csv("D:/ICT/Thesis/Data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv")
train_path = 'D:/ICT/Thesis/Data/rsna-miccai-brain-tumor-radiogenomic-classification/train/'

#%% creating a dataframe to have a metadata form
attributes = ["BraTS21ID", "Sequence Type","PatientName", "StudyInstanceUID"]
SQtypes=['FLAIR', 'T1w', 'T1wCE', 'T2w'] # Sequence tpyes 
df = pd.DataFrame(columns = attributes)

#%% appending the rows of dataframe with attributes
for i in range(len(train_df.index)):
    for types in SQtypes:
        ds = pydicom.dcmread(train_path + '/' + patients[i] + '/' + types + '/' + random.choice(os.listdir(train_path + '/' + patients[i] + '/' + types + '/')))
        df = df.append({'BraTS21ID': train_df.iloc[i]['BraTS21ID'],
                        'Sequence Type': types,
                        'PatientName': patients[i],
                        'StudyInstanceUID': ds.StudyInstanceUID,
                        }
                        ,ignore_index=True)
        

df.to_csv('metadata_test1.csv')