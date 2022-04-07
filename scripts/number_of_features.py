import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.metrics import confusion_matrix
from xgboost import cv
from mlxtend.feature_selection import ExhaustiveFeatureSelector


''' <<to_features>> finds common top i features (by XGBosst classifier) and 
    return them as a dictionary of string which are the name of the features. 
    df: dataframe as an input
    i: number of features needed to be ranked
    
    return: the list of i top ranked features
'''
def top_features(df,i):

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
    im_feat = sorted(feats.items(), key=lambda feats: feats[1], reverse=True)[:i]
    im_feat.sort(key = lambda x: x[1], reverse=True)
    im_feat = [item for sublist in im_feat for item in sublist]
    im_feat = [elm for elm in im_feat if isinstance(elm, str)]
    
        
    # the list of most i-th top ranked features
    return im_feat



''' Splitting the dataset and applying k-fold cross validation
    Feature selection by XGBoost method
    Fitting different model: SVM, LogisticRegression, Random forest, NN
    Changing the numbere of features to see the idieal number
'''
# defining a new empty dataframe to fill with different metrics
metrics = pd.DataFrame(columns=['features_number', 'f1_score_RF', 'accuracy_RF', 'precision_RF', 'recall_RF',
                                 'f1_score_SVM', 'accuracy_SVM', 'precision_SVM', 'recall_SVM',
                                 'f1_score_LR', 'accuracy_LR', 'precision_LR', 'recall_LR',
                                 'f1_score_NN', 'accuracy_NN', 'precision_NN', 'recall_NN'])


data_path = r'D:\ICT\Thesis\Github\repo\data\separateed_data'
data_list = os.listdir(data_path)
data_list = [file.replace('.csv', '') for file in data_list]


    
# reading and splitting the edataset into train and test 
df = pd.read_csv(r'D:\ICT\Thesis\Github\repo\data\all_best_data.csv')

# defining the target value and separate it 
y = df['MGMT_value']
X = df.drop(['MGMT_value', 'Unnamed: 0'], axis = 1)

# getting the top features
list_im_feat = top_features(df, 20)

# dataset with best features
X = X[list_im_feat].copy()

# splitting the whole dataset into train (80%) and test (20%)
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size = 0.2, random_state = 0)

# transform data: final test
scaler = MinMaxScaler()
X_ts = scaler.fit_transform(X_ts)

# convert the test set to the dataframe in order to use it in the while loop
X_ts = pd.DataFrame(X_ts, columns = X_tr.columns)

# applying k-fold cross validation (K=5)
kf = KFold(n_splits=5, shuffle=True)

# iteration over number of features i
i = 20
while i!=0:
    
    # defining accuracy variables
    acc_RF, acc_SVM, acc_LR, acc_NN = [], [], [], []
    f1_RF, f1_SVM, f1_LR, f1_NN = [], [], [], []
    prec_RF, prec_SVM, prec_LR, prec_NN = [], [], [], []
    rec_RF, rec_SVM, rec_LR, rec_NN = [], [], [], []
    
    for train_index , test_index in kf.split(X_tr):
        X_train , X_test = X_tr.iloc[train_index,:], X_tr.iloc[test_index,:]
        y_train , y_test = y_tr.iloc[train_index], y_tr.iloc[test_index]
         
        
        # keep the most top i ranked features
        X_train = X_train[list_im_feat[:i]].copy()
        X_test = X_test[list_im_feat[:i]].copy() 
        X_ts = X_ts[list_im_feat[:i]].copy()
        
        # transform data
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test) 
          
        #%% Random Forest Classifier:
        # Create and train the RandoForestclassifier on the train set
        classifier_RF = RandomForestClassifier()
        classifier_RF.fit(X_train, y_train)
        
        # make a prediction on the validation set and then check model performance by accuracy
        y_pred_RF = classifier_RF.predict(X_train)
        acc_RF.append(accuracy_score(y_train, y_pred_RF))
        # conf_RF.append(confusion_matrix(y_test, y_pred_RF))
        f1_RF.append(f1_score(y_train, y_pred_RF))
        prec_RF.append(precision_score(y_train, y_pred_RF))
        rec_RF.append(recall_score(y_train, y_pred_RF))
        
        #%% Support Vector Machine:
        # build the SVM classifier and train it on the entire training data set
        classifier_SVM = SVC()
        classifier_SVM.fit(X_train, y_train)
        
        # get predictions on the test set and store the accuracy
        y_pred_SVC = classifier_SVM.predict(X_train)
        acc_SVM.append(accuracy_score(y_train, y_pred_SVC))
        # conf_SVM.append(confusion_matrix(y_test, y_pred_SVC))
        f1_SVM.append(f1_score(y_train, y_pred_SVC))
        prec_SVM.append(precision_score(y_train, y_pred_SVC))
        rec_SVM.append(recall_score(y_train, y_pred_SVC))
    
        #%% Logistic Regession:
        # build the classifier and fit the model
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        
        # prediction and store accuracy
        y_pred_LR = logreg.predict(X_train)
        acc_LR.append(accuracy_score(y_train, y_pred_LR))
        # conf_LR.append(confusion_matrix(y_test, y_pred_LR))
        f1_LR.append(f1_score(y_train, y_pred_LR))
        prec_LR.append(precision_score(y_train, y_pred_LR))
        rec_LR.append(recall_score(y_train, y_pred_LR))
        
        #%% Neural Network:
        # create a MLPClassifier and fit the model
        clf = MLPClassifier(solver='lbfgs', 
                        alpha=1e-5,
                        hidden_layer_sizes=(6,), 
                        random_state=1)
    
        clf.fit(X_train, y_train) 
        
        # prediction and store accuracy
        y_pred_NN = clf.predict(X_train)
    
        acc_NN.append(accuracy_score(y_train, y_pred_NN))
        # conf_NN.append(confusion_matrix(y_test, y_pred_NN))
        f1_NN.append(f1_score(y_train, y_pred_NN))
        prec_NN.append(precision_score(y_train, y_pred_NN))
        rec_NN.append(recall_score(y_train, y_pred_NN))
        
    
    # storing result of evaluation metrics in dataframe for furthre anlysis
    # # mean
    # data_to_store = {'features_number': f'{i}' ,'f1_score_RF': np.mean(f1_RF), 'accuracy_RF': np.mean(acc_RF) , 'precision_RF': np.mean(prec_RF), 'recall_RF': np.mean(rec_RF),
    #         'f1_score_SVM': np.mean(f1_SVM), 'accuracy_SVM': np.mean(acc_SVM), 'precision_SVM': np.mean(prec_SVM), 'recall_SVM': np.mean(rec_SVM),
    #         'f1_score_LR': np.mean(f1_LR), 'accuracy_LR': np.mean(acc_LR), 'precision_LR': np.mean(prec_LR), 'recall_LR': np.mean(rec_LR),
    #         'f1_score_NN': np.mean(f1_NN), 'accuracy_NN': np.mean(acc_NN), 'precision_NN': np.mean(prec_NN), 'recall_NN': np.mean(rec_NN)}
    
    # std
    data_to_store = {'features_number': f'{i}' ,'f1_score_RF': np.std(f1_RF), 'accuracy_RF': np.std(acc_RF) , 'precision_RF': np.std(prec_RF), 'recall_RF': np.std(rec_RF),
            'f1_score_SVM': np.std(f1_SVM), 'accuracy_SVM': np.std(acc_SVM), 'precision_SVM': np.std(prec_SVM), 'recall_SVM': np.std(rec_SVM),
            'f1_score_LR': np.std(f1_LR), 'accuracy_LR': np.std(acc_LR), 'precision_LR': np.std(prec_LR), 'recall_LR': np.std(rec_LR),
            'f1_score_NN': np.std(f1_NN), 'accuracy_NN': np.std(acc_NN), 'precision_NN': np.std(prec_NN), 'recall_NN': np.std(rec_NN)}
    
    metrics = metrics.append(data_to_store, ignore_index = True)
    i-=1

# save the data as a csv file
metrics.to_csv(r'D:\ICT\Thesis\Github\repo\results\results_by_std\features_variation_all_train.csv')  
