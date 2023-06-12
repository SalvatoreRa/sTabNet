#!/usr/bin/env python3

print('starting importing')

from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
import xgboost as xgb
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler
import re
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from lifelines.utils import concordance_index
import shap

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


print('starting executing script')

start_time = time.time()
measures = [ 'Accuracy', 'Micro precision', 'Micro recall', 'Micro F1 score',
        'Macro precision', 'Macro recall', 'Macro F1 score',
           'Weighted precision', 'Weighted recall', 'Weighted F1 score',
           'c_index']

#Parameters
n_feat =100
n_inf = 10
n_red =0
n_rep=0
n_classes=6
class_sep = 0.1
#number repetition
n = 100
split_i = np.array(range(n) )
results = pd.DataFrame(index = split_i, columns = measures)

N_feat = np.array(range(n_feat) )
feat_imp = pd.DataFrame(index = N_feat, columns = split_i)

sep_diff = np.round(np.linspace(0.1, 0.9, num=9),1)


for p in sep_diff:
    
    print('separation coefficient:')
    
    print(p)
    
    # CREATE DATASET AND FEATURE IMPORTANCE
    class_sep = p
    # create classification dataset, function from scikit-learn
    # dataset with 1000 examples,10 informative features and 90 non-informative
    # 6 classes
    # p is hyperprarameter controlling separation between classes
    # lower p more difficult classification task
    X, y = make_classification(
        n_samples=1000,
        n_features=n_feat,
        n_informative=n_inf,
        n_redundant=n_red,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep= class_sep,
        random_state=0,
        shuffle = False,
    )
    
    col_names =['col_' + str(i) for i in range(n_feat) ]
    X = pd.DataFrame(X, columns= col_names)

    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    

    #100 different split of the dataset
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                                                random_state=i, stratify = y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, 
                                                          random_state=i, stratify = y_train)
        #scaling the data
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)


            
	#training of a XGB classifier for multi-task objective
        model = xgb.XGBClassifier(objective='multi:softmax', random_state=i)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
	
	# recording of different evaluation metrics for the model
        results.loc[i, "Accuracy"] = accuracy_score(y_test, y_pred)
        results.loc[i, "Micro precision"] = precision_score(y_test, y_pred, average='micro')
        results.loc[i, "Micro recall"] = recall_score(y_test, y_pred, average='micro')
        results.loc[i, "Micro F1 score"] = f1_score(y_test, y_pred, average='micro')
        results.loc[i, "Macro precision"] = precision_score(y_test, y_pred, average='macro')
        results.loc[i, "Macro recall"] = recall_score(y_test, y_pred, average='macro')
        results.loc[i, "Macro F1 score"] = f1_score(y_test, y_pred, average='macro')
        results.loc[i, "Weighted precision"] = precision_score(y_test, y_pred, average='weighted')
        results.loc[i, "Weighted recall"] = recall_score(y_test, y_pred, average='weighted')
        results.loc[i, "Weighted F1 score"] = f1_score(y_test, y_pred, average='weighted')
        
        #storing feature importance for of the model 
        feat_imp.loc[:,i] = model.feature_importances_
        
    res_dir = './results/'
        
    results.to_csv(res_dir+ 'results_XGBoost_inf+' + str(p) +'.csv')
    feat_imp.to_csv(res_dir+ 'feat_imp_XGBoost_inf+' + str(p) +'.csv')
        
    feat_imp['col_feat'] = feat_imp.index
    
    #lineplot of the feature importance for each separate separation coefficient
    feats = feat_imp.melt('col_feat',  var_name='splits', value_name='feat_imp')
    sns_plot =sns.lineplot(data=feats, x="col_feat", y="feat_imp")
    sns_plot.figure.savefig(res_dir+ 'feat_plot_inf+' + str(p) +'.png')
    plt.close()
    
print("--- %s seconds ---" % (time.time() - start_time))

