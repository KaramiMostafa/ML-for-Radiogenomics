import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
from operator import itemgetter


data_path = r'D:\ICT\Thesis\Github\repo\data\t1ce_data'
data_list= os.listdir(data_path)
data_list = [file.replace('.csv', '') for file in data_list] # getting rid of extension 
all_best_data = pd.DataFrame()


for file in data_list:
    
    #%% reading and splitting the edataset into train and test 
    df = pd.read_csv(os.path.join(data_path, file + '.csv'))
    df = df.drop('Unnamed: 0',axis=1)

    # defining the target value and separate it
    y = df['MGMT_value']
    X = df.drop(['MGMT_value','Image'], axis = 1)

    # # splitting the whole dataset into train (80%) and test (20%)
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # defining MSE variables
    MSE_RF, MSE_SVM, MSE_LR, MSE_NN = [], [], [], []

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
        
        # # add the file name for each feature to separate it from the rest
        # list_im_feat[:] = [x + '_' + file  for x in list_im_feat]
        # df.columns = [ 'param' + str(i + 1) for i in range(len(df.columns)) ]
        
        # keep the most top 20 ranked features
        new_X = X[list_im_feat]
        new_X.columns = [ x + '_' + file for x in list(new_X.columns)]
    
    # merge the dataframes together
    all_best_data = pd.concat([all_best_data, new_X], axis=1)

# adding the missing target value and store the new datase
all_best_data = pd.concat([all_best_data, y], axis=1)
all_best_data.to_csv(r'D:\ICT\Thesis\Github\repo\data\all_t1ce_data.csv') 