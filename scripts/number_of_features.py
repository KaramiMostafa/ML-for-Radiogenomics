import pandas as pd
import numpy as np
import os
import seaborn as sns
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from xgboost import cv
from mlxtend.feature_selection import ExhaustiveFeatureSelector

''' <<mean_conf>> simply gets the mean of each element in confiusion matrixs 
    which are the outcome in each k-subset cross validaion.
    
    return: mean of all confision matrics
'''
def mean_conf(confusion_matrix):
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
    
    
''' <<to_features>> finds common top i features (by XGBoost classifier) and 
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


#%% main part
''' Splitting the dataset and applying k-fold cross validation
    Feature selection by XGBoost method
    Fitting different model: SVM, LogisticRegression, Random forest, NN
    Changing the numbere of features to see the idieal number
'''
# defining a new empty dataframe to fill with different metrics
metrics = pd.DataFrame(columns=['features_number','mean_accuracy_NN', 'std_accuracy_NN',
                                'mean_f1score_NN', 'std_f1score_NN', 'confusion_NN',
                                'mean_accuracy_SVM', 'std_accuracy_SVM',
                                'mean_f1score_SVM', 'std_f1score_SVM', 'confusion_SVM',
                                'mean_accuracy_LR', 'std_accuracy_LR',
                                'mean_f1score_LR', 'std_f1score_LR', 'confusion_LR',
                                'mean_accuracy_MLP', 'std_accuracy_MLP',
                                'mean_f1score_MLP', 'std_f1score_MLP', 'confusion_MLP'])

# defining a new empty dataframe for filling best parameters in each iteration
parameters = pd.DataFrame(columns=['features_number','Nearest Neighbor',
                                  'Support Vector Machine', 'Logistic Regresion',
                                  'Multi-layer Perceptron'])

# copy the dataframe for mean and std of metrics
train_metrics = metrics.copy()
test_metrics = metrics.copy()


data_path = r'D:\ICT\Thesis\Github\repo\data\separateed_data'
data_list = os.listdir(data_path)
data_list = [file.replace('.csv', '') for file in data_list]

# reading and splitting the edataset into train and test 
df = pd.read_csv(r'D:\ICT\Thesis\Github\repo\data\all_t2_data.csv')

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

# applying k-fold cross validation (K=10) --> outer loop
cv_outer = KFold(n_splits=10, shuffle=True)

# iteration over number of features i
i = 20
while i!=0:
    
    # defining performance metrics lists for training
    conf_NN_tr, conf_SVM_tr, conf_LR_tr, conf_MLP_tr = [], [], [], []
    acc_NN_tr, acc_SVM_tr, acc_LR_tr, acc_MLP_tr = [], [], [], []
    f1_NN_tr, f1_SVM_tr, f1_LR_tr, f1_MLP_tr = [], [], [], []
    
    # defining performance metrics lists for test
    conf_NN_ts, conf_SVM_ts, conf_LR_ts, conf_MLP_ts = [], [], [], []
    acc_NN_ts, acc_SVM_ts, acc_LR_ts, acc_MLP_ts = [], [], [], []
    f1_NN_ts, f1_SVM_ts, f1_LR_ts, f1_MLP_ts = [], [], [], []
    
    # defining best paramters list to store for traing\test
    best_par_NN, best_par_SVM, best_par_LR, best_par_MLP= [], [], [], []
    
    # configuring thee cross-validation outer loop
    for train_index , test_index in cv_outer.split(X_tr):
        X_train , X_test = X_tr.iloc[train_index,:], X_tr.iloc[test_index,:]
        y_train , y_test = y_tr.iloc[train_index], y_tr.iloc[test_index]
        
        # configuring the cross-validation procedure (inner loop)
        cv_inner = KFold(n_splits=5, shuffle=True, random_state=1)
        
        # keep the most top i ranked features
        X_train = X_train[list_im_feat[:i]].copy()
        X_test = X_test[list_im_feat[:i]].copy() 
        X_ts = X_ts[list_im_feat[:i]].copy()
        
        # transform data
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test) 
          
        #%% Nearst neighbor:
        # Create and train the KNeighborsClassifier on the train\test set
        model_NN = KNeighborsClassifier()
        
        # Set up possible values of parameters to optimize over
        parameters_NN = {'n_neighbors' :[3, 5, 11, 19],
                         'weights':['ubiform', 'distance'],
                         'metric':['euclidean', 'manhattan']}
        
        # define search
        classifier_NN = GridSearchCV(model_NN, parameters_NN, scoring='accuracy', cv=cv_inner, refit=True)
        
        # execute search
        result_NN = classifier_NN.fit(X_train, y_train)
        
        # get the best performing model fit on the whole training\test set + save the best parameters
        best_model_NN = result_NN.best_estimator_
        best_par_NN.append(classifier_NN.best_params_)
        
        # make a prediction on the validation set and then check model performance (train)
        y_pred_NN = best_model_NN.predict(X_train)
        
        acc_NN_tr.append(accuracy_score(y_train, y_pred_NN))
        conf_NN_tr.append(confusion_matrix(y_train, y_pred_NN))
        f1_NN_tr.append(f1_score(y_train, y_pred_NN))
        
        # make a prediction on the validation set and then check model performance (test)
        y_pred_NN = best_model_NN.predict(X_ts)
        
        acc_NN_ts.append(accuracy_score(y_ts, y_pred_NN))
        conf_NN_ts.append(confusion_matrix(y_ts, y_pred_NN))
        f1_NN_ts.append(f1_score(y_ts, y_pred_NN))
        
        #%% Support Vector Machine:
        # build the SVM classifier and train it on the entire training\test data set
        model_SVM = SVC()
        
        # Set up possible values of parameters to optimize over
        parameters_SVM = {'C':[1, 10, 100],'gamma':[0.001, 0.0001]}
        
        # define search
        classifier_SVM = GridSearchCV(model_SVM, parameters_SVM, scoring='accuracy', cv=cv_inner, refit=True)
        
        # execute search
        result_SVM = classifier_SVM.fit(X_train, y_train)
        
        # get the best performing model fit on the whole training\test set + save the best parameters
        best_model_SVM = result_SVM.best_estimator_
        best_par_SVM.append(classifier_SVM.best_params_)
        
        # get predictions on the test set and store the performance metrics (train)
        y_pred_SVC = best_model_SVM.predict(X_train)
        
        acc_SVM_tr.append(accuracy_score(y_train, y_pred_SVC))
        conf_SVM_tr.append(confusion_matrix(y_train, y_pred_SVC))
        f1_SVM_tr.append(f1_score(y_train, y_pred_SVC))
        
        # get predictions on the test set and store the performance metrics (test)
        y_pred_SVC = best_model_SVM.predict(X_ts)
        
        acc_SVM_ts.append(accuracy_score(y_ts, y_pred_SVC))
        conf_SVM_ts.append(confusion_matrix(y_ts, y_pred_SVC))
        f1_SVM_ts.append(f1_score(y_ts, y_pred_SVC))

    
        #%% Logistic Regession:
        # build the classifier and fit the model
        model_LR = LogisticRegression()
        
        # Set up possible values of parameters to optimize over
        parameters_LR = {'C':[0.01, 0.1, 1, 10, 100]}
        
        # define search
        classifier_LR = GridSearchCV(model_LR, parameters_LR, scoring='accuracy', cv=cv_inner, refit=True)
        
        # execute search
        result_LR = classifier_LR.fit(X_train, y_train)
        
        # get the best performing model fit on the whole training set + save the best parameters
        best_model_LR = result_LR.best_estimator_
        best_par_LR.append(classifier_LR.best_params_)
        
        # prediction and store performance metrics (train)
        y_pred_LR = best_model_LR.predict(X_train)
        
        acc_LR_tr.append(accuracy_score(y_train, y_pred_LR))
        conf_LR_tr.append(confusion_matrix(y_train, y_pred_LR))
        f1_LR_tr.append(f1_score(y_train, y_pred_LR))
        
        # prediction and store performance metrics (test)
        y_pred_LR = best_model_LR.predict(X_ts)
        
        acc_LR_ts.append(accuracy_score(y_ts, y_pred_LR))
        conf_LR_ts.append(confusion_matrix(y_ts, y_pred_LR))
        f1_LR_ts.append(f1_score(y_ts, y_pred_LR))

        
        #%% Neural Network:
        # create a MLPClassifier and fit the model
        model_MPL = MLPClassifier(solver='lbfgs', 
                        alpha=1e-5,
                        hidden_layer_sizes=(6,), 
                        random_state=1)
        
        # Set up possible values of parameters to optimize over
        parameters_MLP = {'batch_size': [100, 200],
                          'momentum': [0.9, 0.99 ],
                          'learning_rate_init':[0.001, 0.01, 0.1]}
        
        # define search
        classifier_MLP = GridSearchCV(model_MPL, parameters_MLP, scoring='accuracy', cv=cv_inner, refit=True)
        
        # execute search
        result_MLP = classifier_MLP.fit(X_train, y_train)
        
        # get the best performing model fit on the whole training set + save the best parameters
        best_model_MLP = result_MLP.best_estimator_
        best_par_MLP.append(classifier_MLP.best_params_)
        
        # prediction and store preformance metrics (train)
        y_pred_NN = best_model_MLP.predict(X_train)
    
        acc_MLP_tr.append(accuracy_score(y_train, y_pred_NN))
        conf_MLP_tr.append(confusion_matrix(y_train, y_pred_NN))
        f1_MLP_tr.append(f1_score(y_train, y_pred_NN))
        
        # prediction and store preformance metrics (test)
        y_pred_NN = best_model_MLP.predict(X_ts)
    
        acc_MLP_ts.append(accuracy_score(y_ts, y_pred_NN))
        conf_MLP_ts.append(confusion_matrix(y_ts, y_pred_NN))
        f1_MLP_ts.append(f1_score(y_ts, y_pred_NN))

        
    
    # storing result of evaluation metrics in dataframe for furthre anlysis
    # trainging results
    train_data_to_store = {'features_number':f'{i}' ,'mean_accuracy_NN': np.mean(acc_NN_tr), 'std_accuracy_NN':np.std(acc_NN_tr),
                          'mean_f1score_NN':np.mean(f1_NN_tr) , 'std_f1score_NN':np.std(f1_NN_tr), 'confusion_NN':mean_conf(conf_NN_tr),
                          'mean_accuracy_SVM':np.mean(acc_SVM_tr), 'std_accuracy_SVM':np.std(acc_SVM_tr),
                          'mean_f1score_SVM':np.mean(f1_SVM_tr), 'std_f1score_SVM':np.std(f1_SVM_tr), 'confusion_SVM':mean_conf(conf_SVM_tr),
                          'mean_accuracy_LR':np.mean(acc_LR_tr), 'std_accuracy_LR':np.std(acc_LR_tr),
                          'mean_f1score_LR':np.mean(f1_LR_tr), 'std_f1score_LR':np.std(f1_LR_tr), 'confusion_LR':mean_conf(conf_LR_tr),
                          'mean_accuracy_MLP':np.mean(acc_MLP_tr), 'std_accuracy_MLP':np.std(acc_MLP_tr),
                          'mean_f1score_MLP':np.mean(f1_MLP_tr), 'std_f1score_MLP':np.std(f1_MLP_tr), 'confusion_MLP':mean_conf(conf_MLP_tr)}
    
    # test results
    test_data_to_store = {'features_number':f'{i}' ,'mean_accuracy_NN': np.mean(acc_NN_ts), 'std_accuracy_NN':np.std(acc_NN_ts),
                          'mean_f1score_NN':np.mean(f1_NN_ts) , 'std_f1score_NN':np.std(f1_NN_ts), 'confusion_NN':mean_conf(conf_NN_ts),
                          'mean_accuracy_SVM':np.mean(acc_SVM_ts), 'std_accuracy_SVM':np.std(acc_SVM_ts),
                          'mean_f1score_SVM':np.mean(f1_SVM_ts), 'std_f1score_SVM':np.std(f1_SVM_ts), 'confusion_SVM':mean_conf(conf_SVM_ts),
                          'mean_accuracy_LR':np.mean(acc_LR_ts), 'std_accuracy_LR':np.std(acc_LR_ts),
                          'mean_f1score_LR':np.mean(f1_LR_ts), 'std_f1score_LR':np.std(f1_LR_ts), 'confusion_LR':mean_conf(conf_LR_ts),
                          'mean_accuracy_MLP':np.mean(acc_MLP_ts), 'std_accuracy_MLP':np.std(acc_MLP_ts),
                          'mean_f1score_MLP':np.mean(f1_MLP_ts), 'std_f1score_MLP':np.std(f1_MLP_ts), 'confusion_MLP':mean_conf(conf_MLP_ts)}
    
    # store best parametrs
    par_to_store = {'features_number':f'{i}','Nearest Neighbor':best_par_NN,
                    'Support Vector Machine':best_par_SVM, 'Logistic Regresion':best_par_LR,
                    'Multi-layer Perceptron':best_par_MLP}
    
    train_metrics = train_metrics.append(train_data_to_store, ignore_index = True)
    test_metrics = test_metrics.append(test_data_to_store, ignore_index = True)
    parameters = parameters.append(par_to_store, ignore_index=True)
    
    # reducing number of features for next iteration
    i-=1

# save the data as a csv file
train_metrics.to_csv(r'D:\ICT\Thesis\Github\repo\results\all_mixed_data\features_variation_t2_train.csv')  
test_metrics.to_csv(r'D:\ICT\Thesis\Github\repo\results\all_mixed_data\features_variation_t2_test.csv') 
parameters.to_csv(r'D:\ICT\Thesis\Github\repo\results\all_mixed_data\best_parameters_t2.csv')
