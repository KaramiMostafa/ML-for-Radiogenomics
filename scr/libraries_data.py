"""
libraries : data processing and applying ML algorithms
"""
import os
import pandas as pd
import numpy as np
from sklearn.tree import  DecisionTreeClassifier
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE

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
    
        

    def numeric_data(self,file_name):
        
        """<<numeric_data>>: this function help to eliminate the columns which does not have
            numeric type of data:
                
            parameter --> file (csv) 
            return --> output dataset with only numerical data type
        """
        
        df_raw = pd.read_csv(self.separated_extracted_data_path + file_name)
            
        # excluding the image IDs
        df = df_raw.drop(['Image'], axis = 1)
        
        # excluding all the object types of data + adding back the ID column
        df_numeric = df.select_dtypes(exclude='object')
        df_numeric.insert(0, 'Image' ,df_raw['Image'])
        
        return df_numeric
    

    def low_variance(self,file_name):
        
        """
            <<low_variance>>: this function check feathurs' variance and based on
            the defined threshold, remov the low variance features which give less
            information about the data (the numeric data that has obtained.)
        """
        
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
        
    

    def mean_conf(self, confusion_matrix):
        
        '''
        <<mean_conf>> simply gets the mean of each element in confiusion matrixs 
        which are the outcome in each k-subset cross validaion.
        
        return --> mean of all confision matrics
        '''
        # empty lists to fill up every elements of the different confusion matrics
        e1, e2, e3, e4 = [], [], [], []
        for i in range(0,len(confusion_matrix)):
            e1.append(confusion_matrix[i][0][0])
            e2.append(confusion_matrix[i][0][1])
            e3.append(confusion_matrix[i][1][0])
            e4.append(confusion_matrix[i][1][1])
        # getting mean of each element
        mean_matrix = [[round(np.mean(e1)), round(np.mean(e2))],
                       [round(np.mean(e3)), round(np.mean(e4))]]
        return mean_matrix
    
    def make_confusion_matrix(cf,
                              group_names,
                              categories,
                              title):
        '''
        This function will make a pretty plot of an sklearn Confusion Matrix cm using
        a Seaborn heatmap visualization.
        
        Parameters
        ---------
        cf:            confusion matrix to be passed in
        group_names:   List of strings that represent the labels row by row to be shown in each square.
        categories:    List of strings containing the categories to be displayed on the x,y axis.
        cmap:          Colormap of the values displayed from matplotlib.pyplot.cm.
                       See http://matplotlib.org/examples/color/colormaps_reference.html
        title:         Title for the heatmap.
        
        Returns
        -------
        None
        '''



        group_labels = ["{}\n".format(value) for value in group_names]

        group_percentages = ["{:.2f}".format(value) for value in cf.flatten()/np.sum(cf)]
        group_counts = [f"{round(559*value)}\n" for value in cf.flatten()]


        box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
        box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #Metrics for Binary Confusion Matrices
        precision = cf[1,1] / sum(cf[:,1])
        recall    = cf[1,1] / sum(cf[1,:])
        f1_score  = 2*precision*recall / (precision + recall)
        stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
            accuracy,precision,recall,f1_score)



        # Make the heatmap visualization
        plt.figure(figsize=None)
        sns.heatmap(cf,annot=box_labels,fmt="",cmap='Blues',cbar=True,xticklabels='auto',yticklabels=categories, vmin=0, vmax=1)

        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)

        plt.title(title)
        
    
    def correlation_map(df,list_im_feat,save_path=None, title=None):
        '''
        Parameters
        ----------
        df: DataFrame
            dataframe as an input
        list_im_feat: List
            set of features name extracted from a dataframe (by feature selectors)
        save_path: str
            directory to save the output as a JPG (default is None)
        title: str   
            title for the heatmap (default is None.)

        Returns
        -------
        None.

        '''
        #dividing the whole dataframe into the small part containing only top features
        selected_df = df[list_im_feat].copy()
        
        fig,ax = plt.subplots(figsize=(15,12))
        
        # plotting correlation heatmap
        dataplot = sns.heatmap(selected_df.corr(), cmap="YlGnBu", annot=True)
         
        if save_path!=None and title!=None:
            plt.title(title)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # displaying heatmap
        plt.show()
    
    def top_feat_RFE(self, df, number_feat, target_var):
        '''
            This function finds top features using the Recursive Feature Elimination
            approach and returns a list of str.
            
            Parameter
            ---------
            df: dataframe as an input
            number_feat: number of features needed to be ranked (int)
            target_var: target variable (str)
            
            Return 
            ------
            best_feat: list of top features' name
            
        '''
    
        # defining the target value and separate it 
        y = df[target_var]
        X = df.drop([target_var], axis = 1)
    
        # # define RFECV
        # rfecv = RFECV(estimator=DecisionTreeClassifier(),
        #     cv=StratifiedKFold(5),
        #     scoring="accuracy",
        #     min_features_to_select=number_feat,
        # )
    
        i = number_feat
        best_feat = [] # list of faetures' name
        while i!=0:
        
            rfe = RFE(estimator=DecisionTreeClassifier(), step=1, n_features_to_select=1)
            
            # fit RFE
            rfe.fit(X, y)
            
            top_feat_name = X.iloc[:, int(rfe.get_support(1))].name
            # append the top feature  and remove it for next iteration
            best_feat.append(top_feat_name)
            X.drop(top_feat_name, axis = 1)
            
            i-=1
        
        return best_feat

    
    

    def top_features_XGB(self, df, number_feat):
        
        ''' 
            <<to_features_XGB>> finds common top i features (by XGBoost classifier) and 
            return them as a dictionary of string which are the name of the features. 
            df: dataframe as an input
            i: number of features needed to be ranked
            
            return: the list of i top ranked features
        '''
        
        # defining the target value and separate it
        y = df['MGMT_value']
        X = df.drop(['MGMT_value','Unnamed: 0'], axis = 1)
        
        kf = KFold(n_splits=5, shuffle=True)
        for train_index , test_index in kf.split(X):
            X_train , X_test = X.iloc[train_index,:], X.iloc[test_index,:]
            y_train , y_test = y.iloc[train_index], y.iloc[test_index]
            
            # declare parameters
            params = {
                        'objective':'binary:logistic',
                        'max_depth': 4,
                        'alpha': 10,
                        'learning_rate': 1.0,
                        'n_estimators':100
                    }
            
            # instantiate the classifier 
            xgb_clf = XGBClassifier(**params)
            
            # fit the classifier to the training data
            xgb_clf.fit(X_train, y_train)
            
            # list of features name
            feat_names = list(X_train.columns)
            
            feats = {} # a dict to hold feature_name: feature_importance
            for feature, importance in zip(feat_names, xgb_clf.feature_importances_):
                feats[feature] = importance #add the name/value pair 
            # appending the dictionary of features with their scores by each k subset
            feats.update({x:y for x,y in feats.items() if y!=0})
            
        # sort the features based on their importance
        im_feat = sorted(feats.items(), key=lambda feats: feats[1], reverse=True)[:number_feat]
        # im_feat.sort(key = lambda x: x[1], reverse=True)
        im_feat = [item for sublist in im_feat for item in sublist]
        im_feat = [elm for elm in im_feat if isinstance(elm, str)]
        
        # the list of most i-th top ranked features
        return im_feat
