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