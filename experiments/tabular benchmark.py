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
import shap

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


### Import precedently defined classes
#path should be where the file are saved
import sys  
sys.path.insert(0, "/home/sraieli/py_script/pnet/")

import toy_datasets, tf_layer, tools, matrixes
from tools import keras_cat, history_plot, splitting_data
from tf_layer import LinearGO 


#### Random Walk and sparse matrix generation
import networkx as nx
def random_walk(graph:nx.Graph, node:int, steps:int = 4, p:float=1.0, q:float=1.0):
   """  
   perform a Node2vec random walk for a node in a graph.
   Node2vec random walk is an extension of the random walk, where parameter p and q controls the 
   exploration of local versus a more wide esploration (Breath first versus deep first search)
   code adapted from: https://keras.io/examples/graph/node2vec_movielens/
   
   Parameter
   graph: networkx graph
   p: return parameter, control the likelihood to visit again a node. low value keep local
   q: in-out parameter, inward/outwards node balance; high value local, low value exploration
   node = node in the networkx graph to start the randomwalk
   steps: number of steps
   Return:
   rw: random walk
   
   notes:
   this code works also with isolate nodes, for isolates node, return a random walk of 1, where
   the isolate node is the only node present
   
   example usage:
   rw = random_walk(graph, node, steps, p, q)
   rw = random_walk(G, 0, 4, 1.0, 1.0)
      
   
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
   perform a set of random walks in a networkx graph
   this function is a simple wrapper to perform a set of random walks in a graph
   
   parameters:
   graph: a networkx graph
   rws: number of randomwalks performed for each node (ex: 5, five random walk starting from each node
   will be performed)
   steps: number of steps (visited node) for each random walks
   p: return parameter, control the likelihood to visit again a node. low value keep local
   q: in-out parameter, inward/outwards node balance; high value local, low value exploration
   return:
   a list of random walks
   
   """
   paths = []
   for node in graph.nodes():
     for _ in range(rws):
         paths.append(random_walk(graph, node, steps, p, q))
   return paths

def mapping_rw(rws=None, features=None):
    """mapping clustering labels to a membership matrix
    this is generating the sparse matrix that will be used in the modfied layer
    input
    rws = a list of random walks (as list of list
    features = list of original features
    output
    a panda dataframe where each feature is mapped to the cluster it belongs
    example usage:
    go = mapping_rw(rws=random_walks, features=data.columns.to_list())
    """
    rw_list = [i for i in range(len(random_walks))]
    A = pd.DataFrame(0, columns=rw_list, index=features)
    for i in range(len(random_walks)):
        rw = list(map(int, random_walks[i]))
        for j in rw:
            A.loc[features[j],i] = 1
    return A


##### classification benchmark

n = 10
start_time = time.time()


measures = [ 'Accuracy', "MCC", "cohen_kappa", "balanced_accuracy",  'Precision',
            "Recall","F1", 'AUC','PRC', 'Specificity', 'Sensitivity', 'algorithm', 
            'dataset_name', 'dataset_class']

res_dir = './generalization/'
benchmark= pd.read_csv(res_dir + 'parameter table.txt', sep='\t')
benchmark = benchmark[~benchmark['dataset'].isin(['Higgs', 'road-safety'])]
benchmark = benchmark[benchmark['group'].isin(['clf_num', 'clf_cat'])]
benchmark=benchmark.reset_index(drop=True)

split_i = np.array(range(n) )
results = pd.DataFrame( columns = measures)
for j in range(benchmark.shape[0]):
    

    ratio_clust = benchmark.loc[j, 'ratio_clust']
    batch_size =benchmark.loc[j, 'batch']
    epochs = benchmark.loc[j, 'epochs']
    target = benchmark.loc[j, 'target']
    alg_type = 'sparse net'
    dataset_class = benchmark.loc[j, 'group']
    dataset_name = benchmark.loc[j, 'dataset']

    from datasets import load_dataset
    dataset = load_dataset("inria-soda/tabular-benchmark", 
                           data_files= dataset_class + '/' +dataset_name +'.csv' )


    df =dataset[ "train"].to_pandas()
    data = df.drop([target], axis=1) #discarding ID
    y=  df[target] 

    n = np.linalg.norm(data, axis=0).reshape(1, data.shape[1])
    cosine_mat= data.T.dot(data) / n.T.dot(n)
    cosine_mat =np.where(cosine_mat> .5, 1, np.where(cosine_mat <-0.5, 1, 0))
    np.fill_diagonal(cosine_mat, 0)

    G = nx.Graph(cosine_mat)


    random_walks =get_paths(G, rws= 3, steps= 5)
    go = mapping_rw(rws=random_walks, features=data.columns.to_list())


    n= 10
    for i in range(n):
        X_train, X_test, X_val, y_train_enc, y_val_enc, y_test_enc =splitting_data( data_x = data, 
                                                                                    data_y = y, 
                                                                                    val_split = 0.1, 
                                                                                    test_split = 0.2,
                                                                                    random_seed =i)
        alg_type ='sparse_net'
        tf.keras.backend.clear_session()

        inputs = keras.layers.Input(shape =(X_train.shape[-1],))
        x = attention(mechanism="scaled_dot",bias = False)(inputs)
        x = LinearGO(256, zeroes= go, activation ="tanh")(x)
        x = keras.layers.Dense(64, activation ="relu")(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        x = keras.layers.Dense(y_train_enc.shape[1], activation ="sigmoid")(x)
        model_go = keras.models.Model(inputs, x)
        model_go.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model_go.fit(X_train, y_train_enc,batch_size=batch_size,
            epochs=epochs, verbose=0, validation_data=(X_val, y_val_enc),
        )

        m= classification_metrics(_X_test = X_test, _model = model_go, 
                                 _y_test = y_test_enc, nn= True)
        m =m + [alg_type, dataset_name, dataset_class]
        results.loc[len(results)] = m

        alg_type ='FFNN_att'
        tf.keras.backend.clear_session()
        inputs = keras.layers.Input(shape =(X_train.shape[-1],))
        x = attention(mechanism="scaled_dot")(inputs)
        x = keras.layers.Dense(k, activation ="relu")(x)
        x = keras.layers.Dense(64, activation ="relu")(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        x = keras.layers.Dense(y_train_enc.shape[1], activation ="sigmoid")(x)
        model_go = keras.models.Model(inputs, x)
        model_go.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model_go.fit(X_train, y_train_enc,batch_size=batch_size,
            epochs=epochs, verbose=0, validation_data=(X_val, y_val_enc),
        )

        m= classification_metrics(_X_test = X_test, _model = model_go, 
                                 _y_test = y_test_enc, nn= True)
        m =m + [alg_type, dataset_name, dataset_class]
        results.loc[len(results)] = m

        alg_type ='FFNN'
        tf.keras.backend.clear_session()
        inputs = keras.layers.Input(shape =(X_train.shape[-1],))
        x = keras.layers.Dense(k, activation ="relu")(inputs)
        x = keras.layers.Dense(64, activation ="relu")(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        x = keras.layers.Dense(y_train_enc.shape[1], activation ="sigmoid")(x)
        model_go = keras.models.Model(inputs, x)
        model_go.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model_go.fit(X_train, y_train_enc,batch_size=batch_size,
            epochs=epochs, verbose=0, validation_data=(X_val, y_val_enc),
        )

        m= classification_metrics(_X_test = X_test, _model = model_go, 
                                 _y_test = y_test_enc, nn= True)
        m =m + [alg_type, dataset_name,dataset_class]
        results.loc[len(results)] = m


    for i in range(n):
        y = np.where( df[target] ==df[target].value_counts().index[0], 0, 1)
        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.20, 
                                                                random_state=i, stratify = y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, 
                                                          random_state=i, stratify = y_train)
        #scaling the data
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        #logistic regression
        alg_type ='logistic_regression'
        model = LogisticRegression(random_state=i)
        model.fit(X_train, y_train)
        m= classification_metrics(_X_test = X_test, _model = model, 
                                 _y_test = y_test, nn= False)
        m =m + [alg_type, dataset_name, dataset_class]
        results.loc[len(results)] = m

        #XGBoost
        alg_type ='XGBoost'
        model = xgb.XGBClassifier(random_state=i)
        model.fit(X_train, y_train)
        m= classification_metrics(_X_test = X_test, _model = model, 
                                 _y_test = y_test, nn= False)
        m =m + [alg_type, dataset_name, dataset_class]
        results.loc[len(results)] = m
        
                #XGBoost
        alg_type ='random_forest'
        model = RandomForestClassifier(random_state=i)
        model.fit(X_train, y_train)
        m= classification_metrics(_X_test = X_test, _model = model, 
                                 _y_test = y_test, nn= False)
        m =m + [alg_type, dataset_name, dataset_class]
        results.loc[len(results)] = m

print("--- %s seconds ---" % (time.time() - start_time))
res_dir= './results/'
results.to_csv(res_dir+ 'classification_comparison.csv')