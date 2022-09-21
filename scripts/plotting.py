import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

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
    
#%%
       
# x= df['features_number']
# y= df['mean_accuracy_SVM']
# e= df['std_accuracy_SVM']

# plt.errorbar(x, y, e, linestyle='None', marker='^')
# plt.grid()
# plt.show()
# def comparison_train_test(df_tr, df_ts):
    
# for data_type in ['all','flair','t1','t1ce','t2']:
    
#     df_tr = pd.read_csv('D:\\ICT\\Thesis\\Github\\repo\\results\\' + data_type + '_mixed_data\\features_variation_' + data_type  + '_train.csv')
#     df_ts = pd.read_csv('D:\\ICT\\Thesis\\Github\\repo\\results\\' + data_type + '_mixed_data\\features_variation_' + data_type  + '_test.csv')
    
#     for method in ['NN','LR','SVM','MLP']:
#         x= df_tr['features_number']
        
#         y_tr= df_tr['mean_f1score_' + method]
#         y_ts= df_ts['mean_f1score_' + method]
        
#         e_tr= df_tr['std_f1score_' + method]
#         e_ts= df_ts['std_f1score_' + method]
        
#         result_path = 'D:\\ICT\\Thesis\\Github\\repo\\results\\' + data_type + '_mixed_data\\'
        
#         plt.errorbar(x, y_tr, e_tr, linestyle='None', marker='^', label='Train')
#         plt.errorbar(x, y_ts, e_ts, linestyle='None', marker='^', label='Test')
    
#         plt.xlabel(x.name)
#         plt.ylabel(y_tr.name + '+ std with bar')
        
        
#         plt.title(f'comparison of mean and std of {data_type} mixed data (train\\test)')
#         plt.legend()
#         plt.grid()
#         plt.savefig(result_path + 'f1score_' + method +'_comparison_std&mean.jpg', dpi=300)
#         plt.show()


    
# comparison_train_test(df_tr,df_ts)  



#%%

# for data_type in ['all','flair','t1','t1ce','t2']: 
    
#     result_path = 'D:\\ICT\\Thesis\\Github\\repo\\results\\' + data_type + '_mixed_data\\'
#     df=pd.read_csv('D:\\ICT\\Thesis\\Github\\repo\\results\\' + data_type + '_mixed_data\\features_variation_' + data_type + '_test.csv')   
    
#     for item in ['confusion_NN', 'confusion_LR', 'confusion_MLP', 'confusion_SVM']:
#         for i in range(0,20):
            
#             # plotting and saving the confusion matrix
#             cf_matrix = np.array(json.loads(df[item][i]))
            
#             labels = ['True Neg','False Pos','False Neg','True Pos']
#             categories = ['Zero', 'One']
#             test = make_confusion_matrix(cf_matrix, group_names=labels,
#                                   categories=categories,
#                                   title=f'{item} - data type: {data_type} - number of selected features: {20-i}')
            
#             plt.savefig(result_path + f'{item}' +f'_{20-i}' '.jpg', dpi=150, bbox_inches='tight')
#             plt.close()

#%% correlation heat map
df= pd.read_csv(r'D:\ICT\Thesis\Github\repo\data\all_t2_data.csv')
df2=pd.read_csv(r'D:\ICT\Thesis\Github\repo\results\t2_mixed_data\best_parameters_t2.csv')
list_f = (df2['Best Selected Features'][0])
list_f = ast.literal_eval(list_f)
list_f = [n.strip() for n in list_f]
path = r'D:\ICT\Thesis\Github\repo\results\t2_mixed_data\correlation_heatmap_t2_20features.jpg'
# df3 = df[list_f]
correlation_map(df, list_f, path, title= 'Correlation Heatmap for t2 data')
