#!/usr/bin/env python3

from sklearn.datasets import make_classification
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
import re
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from lifelines.utils import concordance_index
import shap

import sys  
sys.path.insert(0, "/home/sraieli/py_script/pnet/")

import toy_datasets, tf_layer, tools, matrixes
from toy_datasets import random_dataset
from matrixes import concat_go_matrix
from tools import keras_cat, history_plot, splitting_data, list_avg_gene, list_avg_path
from tools import list_avg_path2, compare_nets, recovery_weight
from tf_layer import LinearGO, Linear_dataloop, Linear_dataloop1, attention
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf


import math
import numpy as np 
import pandas as pd
from sklearn.cluster import KMeans
from kneed import KneeLocator
import sklearn.metrics as sklearn_metrics
import matplotlib.pyplot as plt

%matplotlib inline

from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

import networkx as nx
def random_walk(graph:nx.Graph, node:int, steps:int = 4, p:float=1.0, q:float=1.0):
   """  
   perform a Node2vec random walk for a node in a graph 
   p: return parameter, control the likelihood to revisit a node. low value keep local
   q: in-out parameter, inward/outwards node balance; high value local, low value exploration
   """
   if nx.is_isolate(G, node):
        rw = [str(node)]
   else:
       rw = [str(node),]
       start_node = node
       for _ in range(steps):
          weights = []
          neighbors = list(nx.all_neighbors(graph, start_node))
          for neigh in neighbors:
            if str(neigh) == rw[-1]:
                # Control the probability to return to the previous node.
                weights.append(1/ p)
            elif graph.has_edge(neigh,rw[-1]):
                # The probability of visiting a local node.
                weights.append(1)
            else:
                # Control the probability to move forward.
                weights.append(1 / q)

          # we transform the probability to 1
          weight_sum = sum(weights)
          probabilities = [weight / weight_sum for weight in weights]
          walking_node = np.random.choice(neighbors, size=1, p=probabilities)[0]
          rw.append(str(walking_node))
          start_node= walking_node
   return rw

def get_paths(graph:nx.Graph, rws= 10, steps = 4, p=1.0, q=1.0):
   """  
   perform a set of random walks ina graph
   """
   paths = []
   for node in graph.nodes():
     for _ in range(rws):
         paths.append(random_walk(graph, node, steps, p, q))
   return paths

def mapping_rw(rws=None, features=None):
    """mapping clustering labels to a membership matrix
    input
    rws = a list of random walks (as list of list
    features = list of original features
    output
    a panda dataframe where each feature is mapped to the cluster it belongs
    example usage:
    go = mapping_rw(rws=random_walks, features=data.columns.to_list())
    """
    rw_list = [i for i in range(len(rws))]
    A = pd.DataFrame(0, columns=rw_list, index=features)
    for i in range(len(random_walks)):
        rw = list(map(int, rws[i]))
        for j in rw:
            A.loc[features[j],i] = 1
    return A


print('starting executing script')

dataset_name = 'pol'
batch_size =2048
epochs = 500


from datasets import load_dataset
dataset = load_dataset("inria-soda/tabular-benchmark", data_files='clf_num/' +dataset_name + '.csv')
df =dataset[ "train"].to_pandas()
df =dataset[ "train"].to_pandas()
data, y = df.iloc[:, :-1], df.iloc[:, -1]



start_time = time.time()
results = pd.DataFrame( )
accuracy = pd.DataFrame(columns= ['accuracy'] )

n = 100
split_i = np.array(range(n) )
features = data.columns.to_list()
topnodes = pd.DataFrame(0, columns=split_i, index=features )
downnodes = pd.DataFrame(0, columns=split_i, index=features )


for i in range(n):
    print(i)


    X_train, X_test, X_val, y_train_enc, y_val_enc, y_test_enc =splitting_data( data_x = data, 
                                                                                data_y = pd.Series(y), 
                                                                                val_split = 0.1, 
                                                                                test_split = 0.2,
                                                                                random_seed =i)
    
   

    n = np.linalg.norm(X_train, axis=0).reshape(1, X_train.shape[1])
    cosine_mat= X_train.T.dot(X_train) / n.T.dot(n)
    cosine_mat =np.where(cosine_mat> .5, 1, np.where(cosine_mat <-0.5, 1, 0))
    np.fill_diagonal(cosine_mat, 0)
    G = nx.Graph(cosine_mat)
    random_walks =get_paths(G, rws= 3, steps= 5)
    go = mapping_rw(rws=random_walks, features=data.columns.to_list())
    
    tf.keras.backend.clear_session()
    inputs = keras.layers.Input(shape =(X_train.shape[-1],))
    x = attention(mechanism="scaled_dot",bias = False)(inputs)
    x = LinearGO(256, zeroes= go, activation ="tanh")(x)
    x = attention(mechanism="scaled_dot",bias = False)(x)
    x = keras.layers.Dense(64, activation ="relu")(x)
    x = keras.layers.Dropout(rate=0.5)(x)
    x = keras.layers.Dense(y_train_enc.shape[1], activation ="sigmoid")(x)
    model_go = keras.models.Model(inputs, x)
    model_go.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    tf.keras.backend.clear_session()
    history = model_go.fit(
        X_train,
        y_train_enc,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        #callbacks=callbacks,
        validation_data=(X_val, y_val_enc),
        #class_weight=class_weight,
    )
    preds = model_go.predict(X_test)

    y_preds = np.argmax(preds, axis = 1)
    print(accuracy_score(y_test_enc[:,1], y_preds))
    accuracy.loc[str(i),'accuracy'] =accuracy_score(y_test_enc[:,1], y_preds)
    
    
    results.loc[:,str(i)] = model_go.layers[3].weights[1].numpy()
    max_ind = np.argmax(model_go.layers[3].weights[1].numpy())
    rw = [ int(x) for x in random_walks[max_ind] ]
    for j in rw:
        topnodes.loc[features[j],i] = 1
    min_ind = np.argmin(model_go.layers[3].weights[1].numpy())
    rw = [ int(x) for x in random_walks[min_ind] ]
    for j in rw:
        downnodes.loc[features[j],i] = 1

print("--- %s seconds ---" % (time.time() - start_time))
results['mean'] = results.mean(axis=1)
results = results.sort_values('mean', ascending = False)
max_ind = results.index.to_list()[0]
rw = [ int(x) for x in random_walks[max_ind] ]


res_dir = './results/'
results.to_csv(res_dir+ dataset_name + '_att_pat_levels3.csv')
topnodes.to_csv(res_dir+ dataset_name + '_topnodes.csv')
downnodes.to_csv(res_dir+ dataset_name + '_downnodes3.csv')
accuracy.to_csv(res_dir+ dataset_name + '_accuracy3.csv')

##### Ablation ####

import random

remove_list = []

top['presence'] = topnodes.sum(axis=1)/topnodes.shape[1]
top = top.sort_values('presence', ascending = False)
top = top['presence'].iloc[:5]
tops = top.index.to_list()
remove_list.append(tops)


remove_list.append(random.sample([i for i in data.columns.to_list() if i not in [tops][0] ],5))
remove_list.append(random.sample([i for i in data.columns.to_list() if i not in [tops][0] ],5))
print(remove_list)


start_time = time.time()

measures = [ 'Accuracy', 'Specificity', 'Sensitivity', 'TP', 'TN', 'FP', 'FN', 
            'features']

results = pd.DataFrame( columns = measures)
p = list()
for j in range(len(remove_list)):
    X = data.drop(remove_list[j], axis =1)
    
    
    n =10
    
    
    res = pd.DataFrame( columns = measures)
    for i in range(n):
        X_train, X_test, X_val, y_train_enc, y_val_enc, y_test_enc =splitting_data( data_x = X, 
                                                                                data_y = pd.Series(y), 
                                                                                val_split = 0.1, 
                                                                                test_split = 0.2,
                                                                                random_seed =i)
        n = np.linalg.norm(X_train, axis=0).reshape(1, X_train.shape[1])
        cosine_mat= X_train.T.dot(X_train) / n.T.dot(n)
        cosine_mat =np.where(cosine_mat> .5, 1, np.where(cosine_mat <-0.5, 1, 0))
        np.fill_diagonal(cosine_mat, 0)
        start_time = time.time()
        results = pd.DataFrame( )
        accuracy = pd.DataFrame(columns= ['accuracy'] )
        G = nx.Graph(cosine_mat)
        random_walks =get_paths(G, rws= 1, steps= 5)
        go = mapping_rw(rws=random_walks, features=X.columns.to_list())
        
        tf.keras.backend.clear_session()
        inputs = keras.layers.Input(shape =(X_train.shape[-1],))
        x = attention(mechanism="scaled_dot",bias = False)(inputs)
        x = LinearGO(256, zeroes= go, activation ="tanh")(x)
        x = attention(mechanism="scaled_dot",bias = False)(x)
        x = keras.layers.Dense(8, activation ="relu")(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        x = keras.layers.Dense(y_train_enc.shape[1], activation ="sigmoid")(x)
        model_go = keras.models.Model(inputs, x)
        model_go.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

       
        history = model_go.fit(X_train, y_train_enc, batch_size=batch_size,
            epochs=epochs, verbose=0, validation_data=(X_val, y_val_enc),
        )
        preds = model_go.predict(X_test)
        y_preds = np.argmax(preds, axis = 1)
        tn, fp, fn, tp = confusion_matrix(y_test_enc[:,1], y_preds).ravel()
        res.loc[i, "Accuracy"] = accuracy_score(y_test_enc[:,1], y_preds)
        res.loc[i, "Specificity"]  = tn / (tn+fp)
        res.loc[i, "Sensitivity"]  = tp / (tp+fn)
        res.loc[i, "features"]  = j
        res.loc[i, "TN"], res.loc[i, "FP"], res.loc[i, "FN"], res.loc[i, "TP"] =tn, fp, fn, tp
    p.append(res)

results = pd.concat(p)

print("--- %s seconds ---" % (time.time() - start_time))
results = results.reset_index(drop=True)

results.to_csv(res_dir + dataset_name + '_ablation3.csv')