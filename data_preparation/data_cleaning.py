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
''' Cleaning the less usable features.   
'''

#%% Getting the train data labels
df_raw = pd.read_csv(
    r"D:\ICT\Thesis\Github\repo\Pyradiomics\separate_dresult\t2_seg_NCR.csv"
)

# excluding the non-numerical columns
df = df_raw.drop(['MGMT_value','Image'],axis = 1)
df_numeric = df.select_dtypes(exclude='object')
numeric_cols = df_numeric.columns.values

# specifying the columns with low variance (concerning the threshold)
## and dropping those columns
var_thr = VarianceThreshold(threshold = 0.25) #Removing both constant and quasi-constant
var_thr.fit(df_numeric)

concol = [column for column in df_numeric.columns 
          if column not in df_numeric.columns[var_thr.get_support()]]
df_numeric = df_numeric.drop(concol,axis=1)
df_numeric['MGMT_value'] = df_raw['MGMT_value']

df.to_csv('cleaned_t2_seg_NCR.csv')

# # normalizing the whole dataset
# x = df_numeric.values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# df_normalized = pd.DataFrame(x_scaled, columns=df_numeric.columns)

# gettting correlation matrix and sort it
corr_list = df_numeric.corr().unstack().abs()
print(corr_list.sort_values(kind="quicksort")['MGMT_value'])


#%% Recursive Feature Elimination (RFE)

# array = df_numeric.values
# X = array[:,0:(df_numeric.shape[1] - 1)]
# Y = array[:,(df_numeric.shape[1] - 1)]
# # feature extraction
# model = LogisticRegression(max_iter=400)
# rfe = RFE(model, 3)
# fit = rfe.fit(X, Y)

# #Num features
# fit.n_features_

# #selected features
# fit.support_

# #feature ranking
# fit.ranking_


#%% Feature Importance using Decision Tree

# array = df_numeric.values
# X = array[:,0:(df_numeric.shape[1] - 1)]
# Y = array[:,(df_numeric.shape[1] - 1)]
# # feature extraction
# model = DecisionTreeClassifier()
# model.fit(X, Y)
# print(model.feature_importances_) #the smaller the value the best the feature
