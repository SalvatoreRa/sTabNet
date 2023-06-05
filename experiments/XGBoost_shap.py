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

%%capture
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
SHAP_imp = pd.DataFrame(index = N_feat, columns = split_i)

sep_diff = np.round(np.linspace(0.1, 0.9, num=9),1)


for p in sep_diff:
    
    # CREATE DATASET AND FEATURE IMPORTANCE
    class_sep = p
    
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
    
    


    clf = LogisticRegression(random_state=0).fit(X.iloc[:,:n_inf], y)
    c_inf = np.sum(np.abs(clf.coef_), axis=0)
    c_inf = NormalizeData(c_inf) +10
    clf = LogisticRegression(random_state=0).fit(X.iloc[:,n_inf:], y)
    c_noise = np.sum(np.abs(clf.coef_), axis=0)
    c_noise = NormalizeData(c_noise)
    c =list(c_inf) + list(c_noise)
    
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


            #y_val[:5], y_val_enc[:5], y_train[:5],  y_train_enc[:5]

        model = xgb.XGBClassifier(objective='multi:softmax', random_state=i)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

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
        results.loc[i, "c_index"] =concordance_index(c, model.feature_importances_)
        results.loc[i, "c_ind_inf"] =concordance_index(c[:n_inf], model.feature_importances_[:n_inf])
        #results.loc[i, "c_ind_red"] =concordance_index(c[n_inf:(n_inf+n_red)],
        #                                               model.feature_importances_[n_inf:(n_inf+n_red)])
        results.loc[i, "c_ind_Ninf"] =concordance_index(c[n_inf:], 
                                                       model.feature_importances_[n_inf:])
        results.loc[i, "c_ratio"] =np.sum(model.feature_importances_[:n_inf]) / np.sum(model.feature_importances_[n_inf:])
        feat_imp.loc[:,i] = model.feature_importances_
        
        #SHAP
        explainer=shap.Explainer(model)
        shap_values = explainer(pd.DataFrame(X_test, columns =X.columns))
        vals = np.abs(shap_values.values).mean(0)
        vals = np.abs(vals).mean(1)
        feature_names = shap_values.feature_names
        shap_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name', 'feature_importance_vals'])
        #shap_importance.sort_values(by=['col_name'], ascending=True, inplace=True)
        sh =shap_importance.feature_importance_vals.to_list()
        c_index_sh =concordance_index(c, sh)
        c_ratio_sh =np.sum(sh[:n_inf]) / np.sum(sh[n_inf:])
        results.loc[i, "c_index_sh"], results.loc[i, "c_ratio_sh"] = c_index_sh, c_ratio_sh
        SHAP_imp.loc[:,i] = sh
        
    res_dir = './SHAP/'
        
    results.to_csv(res_dir+ 'results_XGBoost_inf+' + str(p) +'.csv')
    feat_imp.to_csv(res_dir+ 'feat_imp_XGBoost_inf+' + str(p) +'.csv')
    SHAP_imp.to_csv(res_dir+ 'SHAP_imp_XGBoost_inf+' + str(p) +'.csv')
        
    feat_imp['col_feat'] = feat_imp.index
    feats = feat_imp.melt('col_feat',  var_name='splits', value_name='feat_imp')
    sns_plot =sns.lineplot(data=feats, x="col_feat", y="feat_imp")
    sns_plot.figure.savefig(res_dir+ 'feat_plot_inf+' + str(p) +'.png')
    plt.close()