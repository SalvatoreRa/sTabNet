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

TGCA
BRCA: breat cancer rnaseq
LUAD: lung cancer rnaseq
'''

df = pd.read_table("/home/sraieli/analysis/brca_metabric/data_expression_median.txt")
df = df.drop([ 'Entrez_Gene_Id'], axis=1)
df1 = df #working on df1 now
df1.index = df1.Hugo_Symbol #renaming the index with gene names
df1 = df1.drop(["Hugo_Symbol"], axis = 1)
df1 = df1.transpose()
df1 = df1.sort_index()
clin = pd.read_table("/home/sraieli/analysis/brca_metabric/data_clinical_patient.txt", skiprows = 4)

clin1 = clin.sort_values(by = ["PATIENT_ID"])
samples = df1.index.tolist()
clin1 = clin1[clin1['PATIENT_ID'].isin(samples)]
clin1 = clin1[clin1['CLAUDIN_SUBTYPE']!= "NC"] #there only 6 examples
df2 = df1[df1.index.isin(clin1['PATIENT_ID'])]
na_col = df2.columns[df2.isna().any()].tolist()
df2 = df2.drop(df2.columns[df2.columns.isin(na_col)], axis=1)
df2 = df2.loc[:,~df2.columns.duplicated()]
df2 = df2.dropna()
meb_subtype =clin1.CLAUDIN_SUBTYPE


go = pd.read_csv("/home/sraieli/dataset/filesforNNarch/filesforNNarch/GOpath_matrix.txt", sep = "\t")
go = go.sort_values("genes")
meb, go2 = genes_go_matrix(gene_df = df2, gene_pat = go, filt_mat = None, 
                   min_max_gene = [10,200], pat_num = 1)


X_train, X_test, X_val, y_train_enc, y_val_enc, y_test_enc =splitting_data( data_x = meb, 
                                                                            data_y = meb_subtype, 
                                                                            val_split = 0.1, 
                                                                            test_split = 0.2,
                                                                            random_seed =42)

#concat exome rnaseq
my_dir = "/home/sraieli/dataset/filesforNNarch/filesforNNarch/"

csv_go = pd.read_csv(my_dir + "go_level.csv", index_col=0)
go = pd.read_csv("/home/sraieli/dataset/filesforNNarch/filesforNNarch/GOpath_matrix.txt", sep = "\t")
go = go.sort_values("genes")

concat, adj_gene, go2= concat_go_matrix(expr = df3, exo_df = cna2, gene_pats =go, filt_mats = None, 
                       min_max_genes = [10,200], pat_nums = 1)

X_train, X_test, X_val, y_train_enc, y_val_enc, y_test_enc =splitting_data( data_x = meb, 
                                                                            data_y = meb_subtype, 
                                                                            val_split = 0.1, 
                                                                            test_split = 0.2,
                                                                            random_seed =42)


#### model trained on metabric

''' this model is trained on METABRIC dataset 
this model is then frozen and used for the following experiments

'''

start_time = time.time()
inputs = keras.layers.Input(shape =(X_train.shape[-1],))
x = attention(mechanism="scaled_dot",bias = False)(inputs)
x = LinearGO(256, zeroes= go2, activation ="tanh")(x)
x = keras.layers.Dense(256, activation ="relu")(x)
x = keras.layers.Dropout(rate=0.3)(x)
x = keras.layers.Dense(64, activation ="relu")(x)
x = keras.layers.Dropout(rate=0.3)(x)
x = keras.layers.Dense(y_train_enc.shape[1], activation ="softmax")(x)
model_METAB= keras.models.Model(inputs, x)
model_METAB.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_METAB.summary()
history = model_METAB.fit(
    X_train,
    y_train_enc,
    batch_size=256,
    epochs=60,
    verbose=0,
    #callbacks=callbacks,
    validation_data=(X_val, y_val_enc),
    #class_weight=class_weight,
)
history_plot(hist = history, mod = model_METAB, cm_m = False, test = X_test,
                  _y = y_test_enc )
print("--- %s seconds ---" % (time.time() - start_time))


preds = model_METAB.predict(X_test)
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test_enc.argmax(axis=1), preds.argmax(axis=1))))

print('Micro Precision: {:.2f}'.format(precision_score(y_test_enc.argmax(axis=1), preds.argmax(axis=1), average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_test_enc.argmax(axis=1), preds.argmax(axis=1), average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test_enc.argmax(axis=1), preds.argmax(axis=1), average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(y_test_enc.argmax(axis=1), preds.argmax(axis=1), average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_test_enc.argmax(axis=1), preds.argmax(axis=1), average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test_enc.argmax(axis=1), preds.argmax(axis=1), average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(y_test_enc.argmax(axis=1), preds.argmax(axis=1), average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_test_enc.argmax(axis=1), preds.argmax(axis=1), average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_test_enc.argmax(axis=1), preds.argmax(axis=1), average='weighted')))
cm = confusion_matrix(preds.argmax(axis=1), y_test_enc.argmax(axis=1))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(4,3))
disp.plot(ax=ax)


##### TGCA-BRCA

''' TCA-BRCA is preprocessed and the model trained before is used 
for feature extraction and fine tuning 
'''

## preprocessing
import pyreadr
res_dir = './TCGA/'
result = pyreadr.read_r(res_dir +'TCGA_BRCA_RNAseq.Rdata')
print(result.keys())
tgca = result['dataFilt']
tgca = np.log2(tgca)
tgca =tgca.replace([np.inf, -np.inf, np.nan], 0)
tgca.columns = [i.split('-')[2] for i in tgca.columns.to_list()]
clin = pd.read_csv(res_dir + 'breast_clinical.tsv', sep = '\t')
clin = clin.drop_duplicates(subset='case_submitter_id', keep="first")
clin = clin[clin['primary_diagnosis'].isin(['Infiltrating duct carcinoma, NOS','Lobular carcinoma, NOS'])]
clin['ID'] = [i.split('-')[2] for i in clin['case_submitter_id']]
clin = clin[clin['ID'].isin(tgca.columns.to_list())]
clin = clin.sort_values(by = ["ID"])
tgca = tgca.transpose()
tgca = tgca.sort_index()
tgca['ID'] = tgca.index
tgca = tgca.drop_duplicates(subset='ID', keep="first")
tgca = tgca.drop('ID', axis =1)
tgca = tgca[tgca.index.isin(clin['ID'])]
subtype_TGCA= clin.primary_diagnosis
gene_list = meb.columns.to_list()
tgca  = tgca.loc[:, tgca.columns.isin(gene_list)]
missing_genes = set(gene_list).difference(set(tgca.columns))
for i in list(missing_genes):
    tgca.loc[:,i] = 0
    
tgca = tgca.reindex(sorted(tgca.columns), axis=1)


X_train[np.isnan(X_train)] = 0
X_test[np.isnan(X_test)] = 0

#### Transfer learning
# model trained on metabric is frozen
model_METAB.summary()
base_model = keras.models.Model(inputs=model_METAB.inputs, outputs=model_METAB.layers[-4].output)
base_model.summary()

'''
model is fine-tuned (just last layers) on TGCA BRCA
'''

measures = ['AUC', 'Accuracy', 'Specificity', 'Sensitivity', 'Precision',
        "Recall", "F1", "balanced_accuracy", "cohen_kappa", 
           "MCC"]
n = 10
split_i = np.array(range(n) )
results = pd.DataFrame(index = split_i, columns = measures)

start_time = time.time()

for i in range(n):
    X_train, X_test, X_val, y_train_enc, y_val_enc, y_test_enc =splitting_data( data_x = tgca, 
                                                                                data_y = tgca_subtype, 
                                                                                val_split = 0.1, 
                                                                                test_split = 0.2,
                                                                                random_seed =i)

    X_train[np.isnan(X_train)] = 0
    X_test[np.isnan(X_test)] = 0
    
    tf.keras.backend.clear_session()
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model(inputs) #or x = base_model(inputs, training=False)
    x = keras.layers.Dense(128, activation ="relu")(x)
    x = keras.layers.Dropout(rate=0.3)(x)
    x = keras.layers.Dense(64, activation ="relu")(x)
    x = keras.layers.Dropout(rate=0.3)(x)
    x = keras.layers.Dense(y_train_enc.shape[1], activation ="sigmoid")(x)
    model_fine_tune = keras.models.Model(inputs, x)
    model_fine_tune.compile(loss='binary_crossentropy', 
                            optimizer=keras.optimizers.Adam(),
                            metrics=['accuracy'])
    
    

    
    history = model_fine_tune.fit(
        X_train,
        y_train_enc,
        batch_size=256,
        epochs=100,
        verbose=0,
        #callbacks=callbacks,
        validation_data=(X_val, y_val_enc),
        #class_weight=class_weight,
    )
    

    
    preds = model_fine_tune.predict(X_test)

    y_preds = np.argmax(preds, axis = 1)

    predictions = preds[:, 1]
    results.loc[i, "Accuracy"] = accuracy_score(y_test_enc[:,1], y_preds)
    results.loc[i, "MCC"] = matthews_corrcoef(y_test_enc[:,1], y_preds)
    results.loc[i, "cohen_kappa"] = cohen_kappa_score(y_test_enc[:,1], y_preds)
    results.loc[i, "balanced_accuracy"] = balanced_accuracy_score(y_test_enc[:,1], y_preds)

    results.loc[i, "AUC"] = roc_auc_score(y_test_enc[:,1],predictions)
    results.loc[i, "Precision"] = precision_score(y_test_enc[:,1], y_preds, average="binary")
    results.loc[i, "Recall"] = recall_score(y_test_enc[:,1], y_preds, average="binary")
    results.loc[i, "F1"] = f1_score(y_test_enc[:,1], y_preds, average="binary")
    tn, fp, fn, tp = confusion_matrix(y_test_enc[:,1], y_preds).ravel()
    results.loc[i, "Specificity"]  = tn / (tn+fp)
    results.loc[i, "Sensitivity"]  = tp / (tp+fn)
print("--- %s seconds ---" % (time.time() - start_time))
res_dir = './results/'
results.to_csv(res_dir+ 'TCGA_BRCA_binary_classif.csv')


#### Feature extraction #####
base_model = keras.models.Model(inputs=model_METAB.inputs, outputs=model_METAB.layers[-4].output)
for layer in base_model.layers:
    layer.trainable = False

X_train, X_test, y_train, y_test = train_test_split(tgca, subtype, test_size=0.20, 
                                                        random_state=i, stratify = subtype)

#scaling the data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train[np.isnan(X_train)] = 0
X_test[np.isnan(X_test)] = 0
    
    
X_train_fts = base_model(X_train)
X_test_fts = base_model(X_test)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train_fts, y_train)
clf.score(X_test_fts, y_test)


measures = ['AUC', 'Accuracy', 'Specificity', 'Sensitivity', 'Precision',
        "Recall", "F1", "balanced_accuracy", "cohen_kappa", 
           "MCC"]
n = 10
split_i = np.array(range(n) )
results = pd.DataFrame(index = split_i, columns = measures)
start_time = time.time()

for i in range(n):
    
    X_train, X_test, y_train, y_test = train_test_split(tgca, subtype, test_size=0.20, 
                                                        random_state=i, stratify = subtype)

    #scaling the data
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    

    X_train[np.isnan(X_train)] = 0
    X_test[np.isnan(X_test)] = 0
    
    X_train_fts = base_model(X_train)
    X_test_fts = base_model(X_test)
    
    clf = LogisticRegression(random_state=0).fit(X_train_fts, y_train)
    

    preds = clf.predict(X_test_fts)
    predictions = clf.predict_proba(X_test_fts)[:, 1]
    results.loc[i, "Accuracy"] = accuracy_score(y_test, preds)
    results.loc[i, "MCC"] = matthews_corrcoef(y_test, preds)
    results.loc[i, "cohen_kappa"] = cohen_kappa_score(y_test, preds)
    results.loc[i, "balanced_accuracy"] = balanced_accuracy_score(y_test, preds)
    results.loc[i, "AUC_precision_recall"] = average_precision_score(y_test, predictions, 
                                                                     pos_label = 'Lobular carcinoma, NOS')
    results.loc[i, "AUC"] = roc_auc_score(y_test,predictions)
    results.loc[i, "Precision"] = precision_score(y_test, preds, average="binary", 
                                                  pos_label = 'Lobular carcinoma, NOS')
    results.loc[i, "Recall"] = recall_score(y_test, preds, pos_label = 'Lobular carcinoma, NOS')
    results.loc[i, "F1"] = f1_score(y_test, preds, pos_label = 'Lobular carcinoma, NOS')
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    results.loc[i, "Specificity"]  = tn / (tn+fp)
    results.loc[i, "Sensitivity"]  = tp / (tp+fn)



print("--- %s seconds ---" % (time.time() - start_time))
res_dir = './TCGA_result/'
results.to_csv(res_dir+ 'TCGA_BRCA_feat_lin_class.csv')


#### TGCA LUAD ####
''' transfer learning for LUAD cancer 
    * fine tuning
    * feature extraction
    similar to what seen above we use the same model trained on the metabric
    model is frozen

'''

### preprocessing
import pyreadr
res_dir = './TCGA/'
result = pyreadr.read_r(res_dir +'TCGA_LUAD_RNAseq.Rdata')
print(result.keys())
tgca = result['dataFilt']
tgca = np.log2(tgca)
tgca =tgca.replace([np.inf, -np.inf, np.nan], 0)
tgca.columns = [i.split('-')[2] for i in tgca.columns.to_list()]
clin = pd.read_csv(res_dir + 'clinical_LUAD.tsv', sep = '\t')

clin['primary_diagnosis'].value_counts()

clin = clin.drop_duplicates(subset='case_submitter_id', keep="first")
clin = clin[clin['primary_diagnosis'].isin(['Adenocarcinoma, NOS',
                                         'Adenocarcinoma with mixed subtypes'])]
clin['ID'] = [i.split('-')[2] for i in clin['case_submitter_id']]
clin = clin[clin['ID'].isin(tgca.columns.to_list())]
clin = clin.sort_values(by = ["ID"])
tgca = tgca.transpose()
tgca = tgca.sort_index()
tgca['ID'] = tgca.index
tgca = tgca.drop_duplicates(subset='ID', keep="first")
tgca = tgca.drop('ID', axis =1)
tgca = tgca[tgca.index.isin(clin['ID'])]
subtype_TGCA= clin.primary_diagnosis

gene_list = meb.columns.to_list()
tgca  = tgca.loc[:, tgca.columns.isin(gene_list)]
missing_genes = set(gene_list).difference(set(tgca.columns))
for i in list(missing_genes):
    tgca.loc[:,i] = 0
tgca = tgca.reindex(sorted(tgca.columns), axis=1)


### finetuning
measures = ['AUC', 'Accuracy', 'Specificity', 'Sensitivity', 'Precision',
        "Recall", "F1", "balanced_accuracy", "cohen_kappa", 
           "MCC"]
n = 10
split_i = np.array(range(n) )
results = pd.DataFrame(index = split_i, columns = measures)

start_time = time.time()

for i in range(n):
    X_train, X_test, X_val, y_train_enc, y_val_enc, y_test_enc =splitting_data( data_x = tgca, 
                                                                                data_y = tgca_subtype, 
                                                                                val_split = 0.1, 
                                                                                test_split = 0.2,
                                                                                random_seed =i)

    X_train[np.isnan(X_train)] = 0
    X_test[np.isnan(X_test)] = 0
    
    tf.keras.backend.clear_session()
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model(inputs) #or x = base_model(inputs, training=False)
    x = keras.layers.Dense(128, activation ="relu")(x)
    x = keras.layers.Dropout(rate=0.3)(x)
    x = keras.layers.Dense(64, activation ="relu")(x)
    x = keras.layers.Dropout(rate=0.3)(x)
    x = keras.layers.Dense(y_train_enc.shape[1], activation ="sigmoid")(x)
    model_fine_tune = keras.models.Model(inputs, x)
    model_fine_tune.compile(loss='binary_crossentropy', 
                            optimizer=keras.optimizers.Adam(learning_rate =0.0001),
                            metrics=['accuracy'])
    
    

    
    history = model_fine_tune.fit(
        X_train,
        y_train_enc,
        batch_size=64,
        epochs=100,
        verbose=0,
        #callbacks=callbacks,
        validation_data=(X_val, y_val_enc),
        #class_weight=class_weight,
    )
    

    
    preds = model_fine_tune.predict(X_test)

    y_preds = np.argmax(preds, axis = 1)

    predictions = preds[:, 1]
    results.loc[i, "Accuracy"] = accuracy_score(y_test_enc[:,1], y_preds)
    results.loc[i, "MCC"] = matthews_corrcoef(y_test_enc[:,1], y_preds)
    results.loc[i, "cohen_kappa"] = cohen_kappa_score(y_test_enc[:,1], y_preds)
    results.loc[i, "balanced_accuracy"] = balanced_accuracy_score(y_test_enc[:,1], y_preds)

    results.loc[i, "AUC"] = roc_auc_score(y_test_enc[:,1],predictions)
    results.loc[i, "Precision"] = precision_score(y_test_enc[:,1], y_preds, average="binary")
    results.loc[i, "Recall"] = recall_score(y_test_enc[:,1], y_preds, average="binary")
    results.loc[i, "F1"] = f1_score(y_test_enc[:,1], y_preds, average="binary")
    tn, fp, fn, tp = confusion_matrix(y_test_enc[:,1], y_preds).ravel()
    results.loc[i, "Specificity"]  = tn / (tn+fp)
    results.loc[i, "Sensitivity"]  = tp / (tp+fn)

print("--- %s seconds ---" % (time.time() - start_time))
res_dir = './results/'
results.to_csv(res_dir+ 'TCGA_LUAD_fine_tuning.csv')


### feature extraction

base_model = keras.models.Model(inputs=model_METAB.inputs, outputs=model_METAB.layers[-4].output)
for layer in base_model.layers:
    layer.trainable = False

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                                        random_state=i, stratify = y)

#scaling the data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train[np.isnan(X_train)] = 0
X_test[np.isnan(X_test)] = 0
    
    
X_train_fts = base_model(X_train)
X_test_fts = base_model(X_test)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train_fts, y_train)
clf.score(X_test_fts, y_test)

measures = ['AUC', 'Accuracy', 'Specificity', 'Sensitivity', 'Precision',
        "Recall", "F1", "balanced_accuracy", "cohen_kappa", 
           "MCC"]
n = 10
split_i = np.array(range(n) )
results = pd.DataFrame(index = split_i, columns = measures)
start_time = time.time()

for i in range(n):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                                        random_state=i, stratify = y)

    #scaling the data
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    

    X_train[np.isnan(X_train)] = 0
    X_test[np.isnan(X_test)] = 0
    
    X_train_fts = base_model(X_train)
    X_test_fts = base_model(X_test)
    
    clf = LogisticRegression(random_state=0).fit(X_train_fts, y_train)
    

    preds = clf.predict(X_test_fts)
    predictions = clf.predict_proba(X_test_fts)[:, 1]
    results.loc[i, "Accuracy"] = accuracy_score(y_test, preds)
    results.loc[i, "MCC"] = matthews_corrcoef(y_test, preds)
    results.loc[i, "cohen_kappa"] = cohen_kappa_score(y_test, preds)
    results.loc[i, "balanced_accuracy"] = balanced_accuracy_score(y_test, preds)
    results.loc[i, "AUC_precision_recall"] = average_precision_score(y_test, predictions, 
                                                                     pos_label = 'Adenocarcinoma, NOS')
    results.loc[i, "AUC"] = roc_auc_score(y_test,predictions)
    results.loc[i, "Precision"] = precision_score(y_test, preds, average="binary", 
                                                  pos_label = 'Adenocarcinoma, NOS')
    results.loc[i, "Recall"] = recall_score(y_test, preds, pos_label = 'Adenocarcinoma, NOS')
    results.loc[i, "F1"] = f1_score(y_test, preds, pos_label = 'Adenocarcinoma, NOS')
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    results.loc[i, "Specificity"]  = tn / (tn+fp)
    results.loc[i, "Sensitivity"]  = tp / (tp+fn)
    
print("--- %s seconds ---" % (time.time() - start_time))
res_dir = './results/'
results.to_csv(res_dir+ 'TCGA_LUAD_feat_lin_class.csv')
