#!/usr/bin/env python3

import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import xgboost as xgb
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler
import optuna
import re
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import sys  
sys.path.insert(0, "/home/sraieli/py_script/pnet/")

import toy_datasets, tf_layer, tools, matrixes
from matrixes import concat_go_matrix
from tools import keras_cat, history_plot, splitting_data
from tf_layer import LinearGO,  attention






##### starting script ####

### preprocessing of the dataset

'''
METABRIC is a multiclass heteromodal dataset
data should download from the cbio portal

file:
data_expression_median: rnaseq data, tabular, float
data_CNA: mutational data, binary data (0-1, where 1 mutation in the gene is present)
data_clinical_patient: meta data about the patient
GO matrix = obtained by the MGSB repository, external knowledge about the features
'''

df = pd.read_table("/home/sraieli/analysis/brca_metabric/data_expression_median.txt")
df = df.drop([ 'Entrez_Gene_Id'], axis=1)
df1 = df #working on df1 now
df1.index = df1.Hugo_Symbol #renaming the index with gene names
df1 = df1.drop(["Hugo_Symbol"], axis = 1)
df1 = df1.transpose()
df1 = df1.sort_index()
clin = pd.read_table("/home/sraieli/analysis/brca_metabric/data_clinical_patient.txt", skiprows = 4)


samples = df1.index.tolist()
clin1 = clin1[clin1['PATIENT_ID'].isin(samples)]
clin1 = clin1[clin1['CLAUDIN_SUBTYPE']!= "NC"] #there only 6 examples
df2 = df1[df1.index.isin(clin1['PATIENT_ID'])]
na_col = df2.columns[df2.isna().any()].tolist()
df2 = df2.drop(df2.columns[df2.columns.isin(na_col)], axis=1)
df2 = df2.loc[:,~df2.columns.duplicated()]
df2 = df2.dropna()
subtype =clin1.CLAUDIN_SUBTYPE
clin1.shape, df2.shape

go = pd.read_csv("/home/sraieli/dataset/filesforNNarch/filesforNNarch/GOpath_matrix.txt", sep = "\t")
go = go.sort_values("genes")
df2, go2 = genes_go_matrix(gene_df = df2, gene_pat = go, filt_mat = None, 
                   min_max_gene = [10,200], pat_num = 1)


#concat exome rnaseq
my_dir = "/home/sraieli/dataset/filesforNNarch/filesforNNarch/"

csv_go = pd.read_csv(my_dir + "go_level.csv", index_col=0)
go = pd.read_csv("/home/sraieli/dataset/filesforNNarch/filesforNNarch/GOpath_matrix.txt", sep = "\t")
go = go.sort_values("genes")

concat, adj_gene, go2= concat_go_matrix(expr = df3, exo_df = cna2, gene_pats =go, filt_mats = None, 
                       min_max_genes = [10,200], pat_nums = 1)


## Test different attention mechanisms


start_time = time.time()
measures = [ 'Accuracy', 'Micro precision', 'Micro recall', 'Micro F1 score',
        'Macro precision', 'Macro recall', 'Macro F1 score',
           'Weighted precision', 'Weighted recall', 'Weighted F1 score']
n = 100


mechs = ["Bahdanau","Graves", "scaled_dot", "Luong"]


for p in mechs:
    
    split_i = np.array(range(n) )
    results = pd.DataFrame(index = split_i, columns = measures)

    for i in range(n):
        X_train, X_test, X_val, y_train_enc, y_val_enc, y_test_enc =splitting_data( data_x = concat, 
                                                                                data_y = subtype, 
                                                                                val_split = 0.1, 
                                                                                test_split = 0.2,
                                                                                random_seed =i)

        tf.keras.backend.clear_session()
        inputs = keras.layers.Input(shape =(concat.shape[-1],))
        x = LinearGO(256, zeroes= adj_gene, activation ="tanh")(inputs)
        x = attention(mechanism=p, bias = False)(x)
        x = LinearGO(256, zeroes= go2, activation ="tanh")(x)
        x = keras.layers.Dense(256, activation ="relu")(x)
        x = keras.layers.Dropout(rate=0.3)(x)
        x = keras.layers.Dense(64, activation ="relu")(x)
        x = keras.layers.Dropout(rate=0.3)(x)
        x = keras.layers.Dense(y_train_enc.shape[1], activation ="softmax")(x)
        model_go = keras.models.Model(inputs, x)
        model_go.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = model_go.fit(
            X_train,
            y_train_enc,
            batch_size=256,
            epochs=60,
            verbose=0,
            #callbacks=callbacks,
            validation_data=(X_val, y_val_enc),
            #class_weight=class_weight,
        )

        preds = model_go.predict(X_test)


        results.loc[i, "Accuracy"] = accuracy_score(y_test_enc.argmax(axis=1), preds.argmax(axis=1))
        results.loc[i, "Micro precision"] = precision_score(y_test_enc.argmax(axis=1), 
                                                            preds.argmax(axis=1), average='micro')
        results.loc[i, "Micro recall"] = recall_score(y_test_enc.argmax(axis=1), 
                                                      preds.argmax(axis=1), average='micro')
        results.loc[i, "Micro F1 score"] = f1_score(y_test_enc.argmax(axis=1), 
                                                    preds.argmax(axis=1), average='micro')
        results.loc[i, "Macro precision"] = precision_score(y_test_enc.argmax(axis=1), 
                                                            preds.argmax(axis=1), average='macro')
        results.loc[i, "Macro recall"] = recall_score(y_test_enc.argmax(axis=1), 
                                                      preds.argmax(axis=1), average='macro')
        results.loc[i, "Macro F1 score"] = f1_score(y_test_enc.argmax(axis=1), 
                                                    preds.argmax(axis=1), average='macro')
        results.loc[i, "Weighted precision"] = precision_score(y_test_enc.argmax(axis=1), 
                                                               preds.argmax(axis=1), average='weighted')
        results.loc[i, "Weighted recall"] = recall_score(y_test_enc.argmax(axis=1),
                                                         preds.argmax(axis=1), average='weighted')
        results.loc[i, "Weighted F1 score"] = f1_score(y_test_enc.argmax(axis=1), 
                                                       preds.argmax(axis=1), average='weighted')

        res_dir = './results/'
        
        results.to_csv(res_dir+ 'METABRIC_multiclass_SparseNet+' + p +'.csv')


