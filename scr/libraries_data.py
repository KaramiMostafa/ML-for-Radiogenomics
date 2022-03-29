"""
libraries : data processing and applying Macjine Learing algorithms
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import VarianceThreshold

""" separated_extracted_data_list: list of separated csv file names
    separated_extracted_data_path: path of separated csv file names
"""
class DataProcess:
    
    def __init__(
        self,
        separated_extracted_data_list,
        separated_extracted_data_path,
        save_path
    ):

        self.separated_extracted_data_list = os.listdir(
            r"D:\ICT\Thesis\Github\out_of_repo\Pyradiomics"
        )
        self.separated_extracted_data_path = r'D:\ICT\Thesis\Github\out_of_repo\Pyradiomics'
        self.save_path = r'D:\ICT\Thesis\Github\repo\data\separateed_data'
    
        
    """<<numeric_data>>: this function help to eliminate the columns which does not have
        numeric type of data:
            input: file (csv) 
            return : output dataset with only numerical data type
    """
    def numeric_data(self,file_name):
        
        df_raw = pd.read_csv(self.separated_extracted_data_path + file_name)
            
        # excluding the image IDs
        df = df_raw.drop(['Image'], axis = 1)
        
        # excluding all the object types of data + adding back the ID column
        df_numeric = df.select_dtypes(exclude='object')
        df_numeric.insert(0, 'Image' ,df_raw['Image'])
        
        return df_numeric
    
    """<<low_variance>>: this function check feathurs' variance and based on
        the defined threshold, remov the low variance features which give less
        information about the data (the numeric data that has obtained.)
        
    """
    def low_variance(self,file_name):
        # getting numerical data with numeric_data function
        df_numeric = DataProcess.numeric_data(file_name)
        
        #Removing both constant and quasi-constant features
        var_thr = VarianceThreshold(threshold = 0.25) 
        var_thr.fit(df_numeric)
        
        concol = [column for column in df_numeric.columns 
                  if column not in df_numeric.columns[var_thr.get_support()]]
        df_numeric = df_numeric.drop(concol,axis=1)
        
        # save the new updated dataset
        df_numeric.to_csv(os.path.join(
            self.save_path,
            'cleaned_'
            +
            file_name
            )
        )
        
    