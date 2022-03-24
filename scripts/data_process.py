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
from sklearn.metrics import accuracy_score
from xgboost import cv
from operator import itemgetter


''' Splitting the dataset and applying k-fold cross validation
    Feature selection by XGBoost method
    Fitting different model: SVM, LogisticRegression, Random forest, NN
'''

#%% reading and splitting the edataset into train and test 
df = pd.read_csv(r'D:\ICT\Thesis\Github\repo\data\separateed_data\cleaned_flair_seg.csv')
df = df.drop('Unnamed: 0',axis=1)

# defining the target value and separate it
y = df['MGMT_value']
X = df.drop(['MGMT_value','Image'], axis = 1)

# # splitting the whole dataset into train (80%) and test (20%)
# X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size = 0.2, random_state = 0)

# defining MSE variables
MSE_XGB, MSE_RF, MSE_SVM, MSE_LR, MSE_NN = [], [], [], [], []



# applying k-fold cross validation (K=5)
kf = KFold(n_splits=5, shuffle=True)

for train_index , test_index in kf.split(X):
    X_train , X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train , y_test = y[train_index], y[test_index]
     

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
    MSE_XGB.append(format(accuracy_score(y_test, y_pred)))
    
    # # k-fold Cross Validation using XGBoost
    # params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1,
    #                 'max_depth': 5, 'alpha': 10}
    
    # xgb_cv = cv(dtrain=data_dmatrix, params=params, nfold=5,
    #                     num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)
    
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
    
    im_feat = dict(sorted(feats.items(), key = itemgetter(1), reverse = True)[:15])
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
    
    # make a prediction on the validation set and then check model performance by MSE
    y_pred_RF = classifier_RF.predict(X_test)
    MSE_RF.append(accuracy_score(y_test, y_pred_RF))
    
    #%% Support Vector Machine:
    # build the SVM classifier and train it on the entire training data set
    classifier_SVM = SVC()
    classifier_SVM.fit(X_train, y_train)
    
    # get predictions on the test set and store the MSE
    y_pred_SVC = classifier_SVM.predict(X_test)
    MSE_SVM.append(accuracy_score(y_test, y_pred_SVC))

    #%% Logistic Regession:
    # build the classifier and fit the model
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    
    # prediction and store MSE
    y_pred_LR = logreg.predict(X_test)
    MSE_LR.append(accuracy_score(y_test, y_pred_LR))
    
    #%% Neural Network:
    # create a MLPClassifier and fit the model
    clf = MLPClassifier(solver='lbfgs', 
                    alpha=1e-5,
                    hidden_layer_sizes=(6,), 
                    random_state=1)

    clf.fit(X_train, y_train) 
    
    # prediction and store MSE
    y_pred_NN = clf.predict(X_test)
    MSE_NN.append(accuracy_score(y_test, y_pred_NN))
    
print('mean of MSE of each method for different k-subset: \n',
'Random Forest: ', np.mean(MSE_RF),
'\n Support Vector Machine: ', np.mean(MSE_SVM),
'\n Logistic Regresiion: ', np.mean(MSE_LR),
'\n MLPClassifier: ', np.mean(MSE_NN),
)
