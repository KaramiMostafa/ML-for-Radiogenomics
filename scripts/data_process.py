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
from sklearn.metrics import confusion_matrix
from xgboost import cv
from operator import itemgetter
from mlxtend.feature_selection import ExhaustiveFeatureSelector



''' Splitting the dataset and applying k-fold cross validation
    Feature selection by XGBoost method
    Fitting different model: SVM, LogisticRegression, Random forest, NN
'''
# defining a new empty dataframe to fill with different metrics
metrics = pd.DataFrame(columns=['Data File','confusion_matrix_RF','f1_score_RF', 'accuracy_RF', 'precision_RF', 'recall_RF',
                                'confusion_matrix_SVM', 'f1_score_SVM', 'accuracy_SVM', 'precision_SVM', 'recall_SVM',
                                'confusion_matrix_LR', 'f1_score_LR', 'accuracy_LR', 'precision_LR', 'recall_LR',
                                'confusion_matrix_NN', 'f1_score_NN', 'accuracy_NN', 'precision_NN', 'recall_NN'])


data_path = r'D:\ICT\Thesis\Github\repo\data\separateed_data'
data_list= os.listdir(data_path)
data_list = [file.replace('.csv', '') for file in data_list]


    
# reading and splitting the edataset into train and test 
df = pd.read_csv(r'D:\ICT\Thesis\Github\repo\data\all_t1ce_data.csv')
# df = df.drop('Unnamed: 0',axis=1)

# defining the target value and separate it
y = df['MGMT_value']
X = df.drop(['MGMT_value'], axis = 1)

# # splitting the whole dataset into train (80%) and test (20%)
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size = 0.2, random_state = 0)

# defining accuracy variables
conf_RF, conf_SVM, conf_LR, conf_NN = [], [], [], []
acc_RF, acc_SVM, acc_LR, acc_NN = [], [], [], []
f1_RF, f1_SVM, f1_LR, f1_NN = [], [], [], []
prec_RF, prec_SVM, prec_LR, prec_NN = [], [], [], []
rec_RF, rec_SVM, rec_LR, rec_NN = [], [], [], []

# applying k-fold cross validation (K=5)
kf = KFold(n_splits=5, shuffle=True)


for train_index , test_index in kf.split(X_tr):
    X_train , X_test = X_tr.iloc[train_index,:], X_tr.iloc[test_index,:]
    y_train , y_test = y_tr.iloc[train_index], y_tr.iloc[test_index]
     

    # XGBoost : defining DMatrix
    data_dmatrix = xgb.DMatrix(data=X,label=y)
    
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
       
    # make predictions on test data
    y_pred = xgb_clf.predict(X_test)
    
    ## compute and print accuracy score
    # print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

    
    ## Feature importance with XGBoost
    # xgb.plot_importance(xgb_clf)
    # plt.figure(figsize = (16, 12))
    # plt.show()
    # print(np.count_nonzero(xgb_clf.feature_importances_), 
    #       np.sort(xgb_clf.feature_importances_))
    
    # list of features name
    feat_names = list(X_train.columns)
    
    feats = {} # a dict to hold feature_name: feature_importance
    for feature, importance in zip(feat_names, xgb_clf.feature_importances_):
        feats[feature] = importance #add the name/value pair 
    feats = {x:y for x,y in feats.items() if y!=0}
    
    # sort the features based on their importance
    im_feat = dict(sorted(feats.items(), key = itemgetter(1), reverse = True)[:20])
    list_im_feat = list(im_feat.keys())
    
    # keep the most top 15 ranked features
    X_train, X_test = X_train[list_im_feat].copy(), X_test[list_im_feat].copy()
    scaler = MinMaxScaler()
    
    
    # transform data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    #%% Random Forest Classifier:
    # Create and train the RandoForestclassifier on the train set
    classifier_RF = RandomForestClassifier()
    classifier_RF.fit(X_train,y_train)
    
    # make a prediction on the validation set and then check model performance by accuracy
    y_pred_RF = classifier_RF.predict(X_test)
    acc_RF.append(accuracy_score(y_test, y_pred_RF))
    conf_RF.append(confusion_matrix(y_test, y_pred_RF))
    f1_RF.append(f1_score(y_test, y_pred_RF))
    prec_RF.append(precision_score(y_test, y_pred_RF))
    rec_RF.append(recall_score(y_test, y_pred_RF))
    
    #%% Support Vector Machine:
    # build the SVM classifier and train it on the entire training data set
    classifier_SVM = SVC()
    classifier_SVM.fit(X_train, y_train)
    
    # get predictions on the test set and store the accuracy
    y_pred_SVC = classifier_SVM.predict(X_test)
    acc_SVM.append(accuracy_score(y_test, y_pred_SVC))
    conf_SVM.append(confusion_matrix(y_test, y_pred_SVC))
    f1_SVM.append(f1_score(y_test, y_pred_SVC))
    prec_SVM.append(precision_score(y_test, y_pred_SVC))
    rec_SVM.append(recall_score(y_test, y_pred_SVC))

    #%% Logistic Regession:
    # build the classifier and fit the model
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    
    # prediction and store accuracy
    y_pred_LR = logreg.predict(X_test)
    acc_LR.append(accuracy_score(y_test, y_pred_LR))
    conf_LR.append(confusion_matrix(y_test, y_pred_LR))
    f1_LR.append(f1_score(y_test, y_pred_LR))
    prec_LR.append(precision_score(y_test, y_pred_LR))
    rec_LR.append(recall_score(y_test, y_pred_LR))
    
    #%% Neural Network:
    # create a MLPClassifier and fit the model
    clf = MLPClassifier(solver='lbfgs', 
                    alpha=1e-5,
                    hidden_layer_sizes=(6,), 
                    random_state=1)

    clf.fit(X_train, y_train) 
    
    # prediction and store accuracy
    y_pred_NN = clf.predict(X_test)

    acc_NN.append(accuracy_score(y_test, y_pred_NN))
    conf_NN.append(confusion_matrix(y_test, y_pred_NN))
    f1_NN.append(f1_score(y_test, y_pred_NN))
    prec_NN.append(precision_score(y_test, y_pred_NN))
    rec_NN.append(recall_score(y_test, y_pred_NN))

# storing result of evaluation metrics in dataframe for furthre anlysis
data_to_store = {'Data File': 'all_best_data' ,'confusion_matrix_RF': conf_RF,'f1_score_RF': np.mean(f1_RF), 'accuracy_RF': np.mean(acc_RF) , 'precision_RF': np.mean(prec_RF), 'recall_RF': np.mean(rec_RF),
       'confusion_matrix_SVM': conf_SVM, 'f1_score_SVM': np.mean(f1_SVM), 'accuracy_SVM': np.mean(acc_SVM), 'precision_SVM': np.mean(prec_SVM), 'recall_SVM': np.mean(rec_SVM),
       'confusion_matrix_LR': conf_LR, 'f1_score_LR': np.mean(f1_LR), 'accuracy_LR': np.mean(acc_LR), 'precision_LR': np.mean(prec_LR), 'recall_LR': np.mean(rec_LR),
       'confusion_matrix_NN': conf_NN, 'f1_score_NN': np.mean(f1_NN), 'accuracy_NN': np.mean(acc_NN), 'precision_NN': np.mean(prec_NN), 'recall_NN': np.mean(rec_NN)}
metrics = metrics.append(data_to_store, ignore_index = True)

# save the data as a csv file
metrics.to_csv(r'D:\ICT\Thesis\Github\repo\results\metrics_all_t1ce_data.csv')     

# for file in data_list:
    
#     # reading and splitting the edataset into train and test 
#     df = pd.read_csv(os.path.join(data_path, file + '.csv'))
#     df = df.drop('Unnamed: 0',axis=1)
    
#     # defining the target value and separate it
#     y = df['MGMT_value']
#     X = df.drop(['MGMT_value','Image'], axis = 1)
    
#     # # splitting the whole dataset into train (80%) and test (20%)
#     X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
#     # defining accuracy variables
#     conf_RF, conf_SVM, conf_LR, conf_NN = [], [], [], []
#     acc_RF, acc_SVM, acc_LR, acc_NN = [], [], [], []
#     f1_RF, f1_SVM, f1_LR, f1_NN = [], [], [], []
#     prec_RF, prec_SVM, prec_LR, prec_NN = [], [], [], []
#     rec_RF, rec_SVM, rec_LR, rec_NN = [], [], [], []
    
#     # applying k-fold cross validation (K=5)
#     kf = KFold(n_splits=5, shuffle=True)
    
    
#     for train_index , test_index in kf.split(X_tr):
#         X_train , X_test = X_tr.iloc[train_index,:], X_tr.iloc[test_index,:]
#         y_train , y_test = y_tr.iloc[train_index], y_tr.iloc[test_index]
         
    
#         # XGBoost : defining DMatrix
#         data_dmatrix = xgb.DMatrix(data=X,label=y)
        
#         # declare parameters
#         params = {
#                     'objective':'binary:logistic',
#                     'max_depth': 4,
#                     'alpha': 10,
#                     'learning_rate': 1.0,
#                     'n_estimators':100
#                 }
        
#         # instantiate the classifier 
#         xgb_clf = XGBClassifier(**params)
        
#         # fit the classifier to the training data
#         xgb_clf.fit(X_train, y_train)
           
#         # make predictions on test data
#         y_pred = xgb_clf.predict(X_test)
        
#         ## compute and print accuracy score
#         # print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
    
        
#         ## Feature importance with XGBoost
#         # xgb.plot_importance(xgb_clf)
#         # plt.figure(figsize = (16, 12))
#         # plt.show()
#         # print(np.count_nonzero(xgb_clf.feature_importances_), 
#         #       np.sort(xgb_clf.feature_importances_))
        
#         # list of features name
#         feat_names = list(X_train.columns)
        
#         feats = {} # a dict to hold feature_name: feature_importance
#         for feature, importance in zip(feat_names, xgb_clf.feature_importances_):
#             feats[feature] = importance #add the name/value pair 
#         feats = {x:y for x,y in feats.items() if y!=0}
        
#         # sort the features based on their importance
#         im_feat = dict(sorted(feats.items(), key = itemgetter(1), reverse = True)[:15])
#         list_im_feat = list(im_feat.keys())
        
#         # keep the most top 15 ranked features
#         X_train, X_test = X_train[list_im_feat].copy(), X_test[list_im_feat].copy()
#         scaler = MinMaxScaler()
        
        
#         # transform data
#         X_train = scaler.fit_transform(X_train)
#         X_test = scaler.fit_transform(X_test)
        
#         #%% Random Forest Classifier:
#         # Create and train the RandoForestclassifier on the train set
#         classifier_RF = RandomForestClassifier()
#         classifier_RF.fit(X_train,y_train)
        
#         # make a prediction on the validation set and then check model performance by accuracy
#         y_pred_RF = classifier_RF.predict(X_test)
#         acc_RF.append(accuracy_score(y_test, y_pred_RF))
#         conf_RF.append(confusion_matrix(y_test, y_pred_RF))
#         f1_RF.append(f1_score(y_test, y_pred_RF))
#         prec_RF.append(precision_score(y_test, y_pred_RF))
#         rec_RF.append(recall_score(y_test, y_pred_RF))
        
#         #%% Support Vector Machine:
#         # build the SVM classifier and train it on the entire training data set
#         classifier_SVM = SVC()
#         classifier_SVM.fit(X_train, y_train)
        
#         # get predictions on the test set and store the accuracy
#         y_pred_SVC = classifier_SVM.predict(X_test)
#         acc_SVM.append(accuracy_score(y_test, y_pred_SVC))
#         conf_SVM.append(confusion_matrix(y_test, y_pred_SVC))
#         f1_SVM.append(f1_score(y_test, y_pred_SVC))
#         prec_SVM.append(precision_score(y_test, y_pred_SVC))
#         rec_SVM.append(recall_score(y_test, y_pred_SVC))
    
#         #%% Logistic Regession:
#         # build the classifier and fit the model
#         logreg = LogisticRegression()
#         logreg.fit(X_train, y_train)
        
#         # prediction and store accuracy
#         y_pred_LR = logreg.predict(X_test)
#         acc_LR.append(accuracy_score(y_test, y_pred_LR))
#         conf_LR.append(confusion_matrix(y_test, y_pred_LR))
#         f1_LR.append(f1_score(y_test, y_pred_LR))
#         prec_LR.append(precision_score(y_test, y_pred_LR))
#         rec_LR.append(recall_score(y_test, y_pred_LR))
        
#         #%% Neural Network:
#         # create a MLPClassifier and fit the model
#         clf = MLPClassifier(solver='lbfgs', 
#                         alpha=1e-5,
#                         hidden_layer_sizes=(6,), 
#                         random_state=1)
    
#         clf.fit(X_train, y_train) 
        
#         # prediction and store accuracy
#         y_pred_NN = clf.predict(X_test)

#         acc_NN.append(accuracy_score(y_test, y_pred_NN))
#         conf_NN.append(confusion_matrix(y_test, y_pred_NN))
#         f1_NN.append(f1_score(y_test, y_pred_NN))
#         prec_NN.append(precision_score(y_test, y_pred_NN))
#         rec_NN.append(recall_score(y_test, y_pred_NN))

#     # storing result of evaluation metrics in dataframe for furthre anlysis
#     data_to_store = {'Data File': file ,'confusion_matrix_RF': conf_RF,'f1_score_RF': np.mean(f1_RF), 'accuracy_RF': np.mean(acc_RF) , 'precision_RF': np.mean(prec_RF), 'recall_RF': np.mean(rec_RF),
#            'confusion_matrix_SVM': conf_SVM, 'f1_score_SVM': np.mean(f1_SVM), 'accuracy_SVM': np.mean(acc_SVM), 'precision_SVM': np.mean(prec_SVM), 'recall_SVM': np.mean(rec_SVM),
#            'confusion_matrix_LR': conf_LR, 'f1_score_LR': np.mean(f1_LR), 'accuracy_LR': np.mean(acc_LR), 'precision_LR': np.mean(prec_LR), 'recall_LR': np.mean(rec_LR),
#            'confusion_matrix_NN': conf_NN, 'f1_score_NN': np.mean(f1_NN), 'accuracy_NN': np.mean(acc_NN), 'precision_NN': np.mean(prec_NN), 'recall_NN': np.mean(rec_NN)}
#     metrics = metrics.append(data_to_store, ignore_index = True)

# # save the data as a csv file
# metrics.to_csv(r'D:\ICT\Thesis\Github\repo\results\metrics_separated.csv') 