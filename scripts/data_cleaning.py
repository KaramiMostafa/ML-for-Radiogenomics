import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
import os

''' Cleaning thee metadata by removing object type and applying
    low variance method to eliminate unusable features.   
'''

#%% Getting the train data labels
data_list= os.listdir(
    r"D:\ICT\Thesis\Github\incase\Pyradiomics\separate_dresult"
)
data_path = r'D:\ICT\Thesis\Github\incase\Pyradiomics\separate_dresult'
save_path = r'D:\ICT\Thesis\Github\repo\data\separateed_data'

for file in data_list:

    df_raw = pd.read_csv(os.path.join(
        data_path,
        file
        )
    )
    
    # excluding the non-numerical columns
    df = df_raw.drop(['MGMT_value','Image'], axis = 1)
    
    df_numeric = df.select_dtypes(exclude='object')
    numeric_cols = df_numeric.columns.values
    
    # specifying the columns with low variance (concerning the threshold)
    ## and dropping those columns
    var_thr = VarianceThreshold(threshold = 0.25) #Removing both constant and quasi-constant
    var_thr.fit(df_numeric)
    
    concol = [column for column in df_numeric.columns 
              if column not in df_numeric.columns[var_thr.get_support()]]
    df_numeric = df_numeric.drop(concol,axis=1)
    df_numeric.insert(0, 'Image' ,df_raw['Image'])
    df_numeric.insert(1, 'MGMT_value' ,df_raw['MGMT_value'])

    df_numeric.to_csv(os.path.join(
        save_path,
        'cleaned_'
        +
        file
        )
    )
