from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import pandas as pd
import seaborn as sns
import gseapy as gp
from scipy.stats import t
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def keras_cat(target, categories):
    """ 
    take target variable and transforme it in array for keras
    perform one-hot encoding
    target = target variables
    categories = the categories available in the target
    output = an np.array, each column a categories, 0-1 for each categories
    """
    target = target.tolist()
    mapping = {}
    for x in range(len(categories)):
      mapping[categories[x]] = x
    for x in range(len(target)):
      target[x] = mapping[target[x]]
    one_hot_encode = to_categorical(target)
    return one_hot_encode



def history_plot(hist = None, cm_m = True, mod = None, test = None, _y =  None):
    '''
    plot history loss, accuracy and confusion matrix (optional)
    required
        history = keras history
    optional
        keras model = classifier
        test = test dataset
        y encoded = the encoded y 
    example usage
    #history_plot(hist = history, mod = model_go, cm_m = True, test = X_test,
                  _y = y_test_enc )
    #history_plot(history, True, model_go,  X_test,  y_test_enc )
    '''
    print(hist.history.keys())
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.show()
    # summarize history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    
    if cm_m:
        #optional
        _preds = mod.predict(test)
        _y_preds = np.argmax(_preds, axis = 1)
        cm = confusion_matrix(_y[:,1], _y_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()

        

def recovery_weight(_layer  = None, annot = False, path_mat= None):
    '''
    recovery_weight v2
    take a linera_go layer as input and it is recovery the weights
    input
    take the pathway matrix the linear_layer 
    return
    the weight, the bias and the sparse matrix
    optional
    if annot true, return the annotation of the matrix in gather format
    example usage:
    w, b, wm = recovery_weight(_layer  = linear_layer, annot = True, path_mat= go )
    '''
    _w = _layer.weights[0].numpy()
    _b = _layer.weights[1].numpy()
    
    if annot:
        print("annotated matrix/ True")
        _b = pd.Series(_b, index = path_mat.columns)
        sparse_mat = tf.convert_to_tensor(path_mat, dtype=tf.float32)
        idx_sparse = tf.where(tf.not_equal(sparse_mat, 0))
        sparse = tf.SparseTensor(idx_sparse, _w, sparse_mat.get_shape())
        _wm = tf.sparse.to_dense(sparse).numpy()
        _wm = pd.DataFrame(_wm, index = path_mat.index,  columns = path_mat.columns).T
        _wm = pd.melt(_wm.reset_index(), id_vars='index')
        _wm = _wm[_wm.value != 0]
        _wm.columns =  ["to_node", "from_node", "weight_value"]
        
    return _w, _b, _wm

def list_avg_gene(_dir = None, _file = None, _n =20, y= 'genes'):
    '''
    return the plot of the top20 in absolute value, the csv with the average
    INPUT
    _dir = where to load and to save the file
    _file = the file where are stored data from a loop
    _N = numbero of genes to plot
    y = is for the column names for plotting, default is "genes" for genes
    OUTPUT
    bar plot
    csv with genes and average
    EXAMPLE USAGE
    list_avg_gene(_file_dir = "/home/sraieli/test/go_test/", 
              _file = "goconcat_net2_genes_mut.csv", _n =20)
    list_avg_gene(_dir =  "/home/sraieli/test/path_test/", 
              _file = "goconcat_net4_path.csv", _n =20, y = "pathway")
    '''
    _df = pd.read_csv(_dir + _file,  index_col=0)
    _df1 = _df.dropna(thresh=2)
    _df1["avg"] = _df1.iloc[:,1:_df1.shape[1]].mean(axis= 1)
    _df1 =_df1.reindex(_df1.avg.abs().sort_values(ascending=False).index)
    _df2 = _df1.iloc[0:_n,:].copy()
    ax = sns.barplot(data = _df2, x = 'avg', y = y)
    fig = ax.get_figure()
    _f = _file.split(".csv")
    fig.savefig(_dir +_f[0] +"_top_" + str(_n) + ".png", dpi = 300, bbox_inches='tight') 
    plt.close()
    _df2 = _df1[[y, "avg"]]
    _df2.to_csv(_dir +_f[0] + "_avg.csv")




def compute_corrected_ttest(serie1, serie2, dsize = [144,40]):
    """Computes right-tailed paired t-test with corrected variance
    using Corrects standard deviation using Nadeau and Bengio's approach
    serie1 = ndarray of shape or list
    serie2 = ndarray of shape or list
    dsize = size of train and test et
    dsize 1= y_train size, for calculate number of observation
    dsize 2 = y_test size, for calculate number of observation
    https://scikit-learn.org/0.24/auto_examples/model_selection/plot_grid_search_stats.html
    """
    differences = [y - x for y, x in zip(serie1, serie2)]
    n_train =dsize[0]
    n_test =dsize[1]
    n = n_train + n_test
    df = n -1
    
    corrected_var = (
        np.var(differences, ddof=1) * ((1 / n) + (n_test / n_train))
    )
    corrected_std = np.sqrt(corrected_var)
    
    mean = np.mean(differences)
    t_stat = mean / corrected_std
    p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
    return t_stat, p_val




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def splitting_data( data_x = None, data_y = None, val_split = 0.1, test_split = 0.2,
                  random_seed =42):

    """ splitting dataset in train, val, test
    this function take a dataset and it is splitting in three parts
    data are returning scaled (data leakage is avoided)
    category variable are returne encoded
    this is for classification dataset
    INPUT
    data_x = matrix (or dataframe), this the X dataset, a matrix
    data_y = a panda df (with only one series, or it can be a series)
    random_seed = the random seed, default is 42
    val_split = percentage of split of validation set , default = 0.1
    test_split = percentage of split of test set , default = 0.2
    OUTPUT
    _X_train, _X_test, _X_val = splitted data_x, data are independently scaled
    _y_train_enc, _y_val_enc, _y_test_enc = encoded y variable
    EXAMPLE OF USAGE
    X_train, X_test, X_val, y_train_enc, y_val_enc, y_test_enc =splitting_data( data_x = concat, 
                                                                            data_y = outcome, 
                                                                            val_split = 0.1, 
                                                                            test_split = 0.2,
                                                                            random_seed =42)
    
    """
    
    _X_train, _X_test, _y_train, _y_test = train_test_split(data_x, data_y, 
                                                            test_size=test_split, 
                                                            random_state=random_seed, 
                                                            stratify = data_y)
    _X_train, _X_val, _y_train, _y_val = train_test_split(_X_train, _y_train, 
                                                          test_size=val_split, 
                                                          random_state=random_seed, 
                                                          stratify = _y_train)
    scaler = MinMaxScaler()
    scaler.fit(_X_train)
    _X_train = scaler.transform(_X_train)
    _X_val = scaler.transform(_X_val)
    _X_test = scaler.transform(_X_test)
    try:
        tar_categ = data_y.iloc[:,0].value_counts().index.tolist()
        _y_train_enc = keras_cat( _y_train.iloc[:,0], tar_categ)
        _y_val_enc = keras_cat( _y_val.iloc[:,0],tar_categ)
        _y_test_enc = keras_cat( _y_test.iloc[:,0],tar_categ)
    except:
        print('pd.series detected')
        tar_categ = False
    finally:
        if tar_categ == False:
            tar_categ = data_y.value_counts().index.tolist()
            _y_train_enc = keras_cat( _y_train, tar_categ)
            _y_val_enc = keras_cat( _y_val, tar_categ)
            _y_test_enc = keras_cat( _y_test, tar_categ)
    
    return _X_train, _X_test, _X_val, _y_train_enc, _y_val_enc, _y_test_enc
    
 
   
def classification_metrics(_X_test = None, _model = None, _y_test = None, nn= True):
    '''
    simple wrapper for getting all the classification metrics
    used for binary classsification
    
    ----
    Parameters:
    input
    - test set, panda dataframe or numpy array
    - model, a trained model
    - y_test, the target variable,  panda series or numpy array or a list
    -nn, bool, if it is a neural network or scikit-learn like algorithm
    
    ----
    Output
    - a list of classification metrics result
    
    ---
    example of usage:
    classification_metrics(_X_test = X_test, _model = model_go, _y_test = y_test_enc, nn= True)
    classification_metrics(_X_test = X_test, _model = model, _y_test = y_test, nn= False)
    '''
    if nn:
        preds = _model.predict(_X_test)  
        y_preds = np.argmax(preds, axis = 1)
        predictions = preds[:, 1]
        _y_test =_y_test[:,1]
        
    else:
        y_preds = _model.predict(_X_test)
        predictions = _model.predict_proba(X_test)[:, 1]
        
        
    m =  [] 
    ms = [accuracy_score, matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score, 
              precision_score, recall_score, f1_score ] 
    for i in range(len(ms)):
        m.append(ms[i](_y_test, y_preds))

    m.append(roc_auc_score(_y_test,predictions))
    m.append(average_precision_score(_y_test, predictions))

    tn, fp, fn, tp = confusion_matrix(_y_test, y_preds).ravel()
    m.append(tn / (tn+fp))
    m.append(tp / (tp+fn))
    return m
    

import networkx as nx
def random_walk(graph:nx.Graph, node:int, steps:int = 4, p:float=1.0, q:float=1.0):
   """  
   perform a Node2vec random walk for a node in a graph.
   Node2vec random walk is an extension of the random walk, where parameter p and q controls the 
   exploration of local versus a more wide esploration (Breath first versus deep first search)
   code adapted from: https://keras.io/examples/graph/node2vec_movielens/
   
   ----
   Parameter
   input
   graph: networkx graph
   p: return parameter, control the likelihood to visit again a node. low value keep local
   q: in-out parameter, inward/outwards node balance; high value local, low value exploration
   node = node in the networkx graph to start the randomwalk
   steps: number of steps
   
   ----
   Return:
   rw: random walk
   
   ----
   notes:
   this code works also with isolate nodes, for isolates node, return a random walk of 1, where
   the isolate node is the only node present
   
   ----
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
   
   
   example usage:
   random_walks =get_paths(G, rws= 1, steps= 5)
   
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
    
def generate_feature_graph(data:pd.DataFrame, metric:str='cosine', score:float=0.5):
    
    '''
    generate a feature graph from a pandas data frame.
    starting from the data matrix or a numpy array a similarity matrix (using a metric)
    is calculated if the similarity is higher than a certain value (score)
    
    ----
    Parameters
    Input
    -data: pandas data frame or numpy array
    -metric: a metric you use to generate the similarity matrix, option are:
        'cosine', 'correlation'. 'cosine' is the defaul
        cosine: cosine similarity, cosine similarity is calculated.
        correlation: pearson correlation
    
    optional arguments
    -score: 0.5 is default, value of similarity for defining an edge
    
    ----
    output
    return a feature graph (networkx graph) which can contains isolated nodes
    
    ----
    example usage
    G =generate_feature_graph(data)
    G =generate_feature_graph(data, metric ='correlation')
    G =generate_feature_graph(data, metric ='correlation')
    G =generate_feature_graph(data, metric ='correlation', score= 0.7)
    
    '''
    if metric =='cosine':
        n = np.linalg.norm(data, axis=0).reshape(1, data.shape[1])
        dist_matrix= data.T.dot(data) / n.T.dot(n)
        dist_matrix =np.where(dist_matrix> score, 1, np.where(dist_matrix <-score, 1, 0))
        np.fill_diagonal(dist_matrix, 0)
    if metric =='correlation':
        dist_matrix =np.corrcoef(data, rowvar= False)
        dist_matrix =np.where(dist_matrix> score, 1, np.where(dist_matrix <-score, 1, 0))
        np.fill_diagonal(dist_matrix, 0)
        
    G = nx.Graph(dist_matrix)
    return G




