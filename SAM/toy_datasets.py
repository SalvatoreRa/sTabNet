import numpy as np
import random
from tensorflow.keras.utils import to_categorical
import pandas as pd

def random_dataset(pat = 500, genes =100, pathway = 50):
    '''
    create a random dataset to test linear pathway
    parameter, number of patients, number of genes, number of pathway
    return: 
    X = dataframe with row number (n patients), column (n genes), 
        dummy names as column and index
    go = dataframe with row number (n genes), column (n pathway), 
        dummy names as column and index
    y_enc = y categorically encoded,number of patients
    EXAMPLE USAGE
    X, y, go = random_dataset(pat = 500, genes =100, pathway = 50)
    '''
    _X = np.random.rand(pat,genes)
    _y = []
    for i in range(0,pat):
        n = random.randint(0,1)
        _y.append(n)
    _y_enc = to_categorical(_y)
    _go =np.random.randint(2, size=(genes, pathway))
    _go = pd.DataFrame(_go, columns = ["go" +str(i) for i in range(1,_go.shape[1] +1,1)],
                     index = ["bla" +str(i) for i in range(1,_go.shape[0] +1,1)])
    _X = pd.DataFrame(_X, columns = ["bla" +str(i) for i in range(1,_X.shape[1] +1,1)],
                     index = ["pat" +str(i) for i in range(1,_X.shape[0] +1,1)])
    return _X, _y_enc, _go

def random_go(go_m = None, ones = "same"):
    '''
    generate a random go matrix equivalent to a go matrix
    generates a sparse matrix similar a gene ontology matrix
    INPUT
    go_m: a go matrix, pathway on top as column name
          gene as indexes name, 1 for connection
    ones: the quantity of ones
          "random", totally random
          "same", default, equal number of 1 of the original matrix
          the number of index is random and columns is random,
          the distribution of the ones in the conums is the same, just random shuffles
    OUTPUT
    A random go matrix, keep seem indexes and pathway name
    EXAMPLE USAGE
    rand_go = random_go(go_m = go2, ones = "same")
    '''
    if ones == "same":
        print("same number of 1")
        randGO  =np.random.randint(1, size=(go_m.shape[0], go_m.shape[1]))
        randGO = pd.DataFrame(randGO, columns = go_m.columns,
                     index = go_m.index)
        _n = go_m.sum().to_list()
        random.shuffle(_n)
        for i in range(go_m.shape[1]):
            idx =random.sample(range(randGO.shape[0]), _n[i])
            for j in idx:
                randGO.iloc[j,i] = 1
    if ones == "random":
        print("random number of 1")
        randGO  =np.random.randint(2, size=(go_m.shape[0], go_m.shape[1]))
        randGO = pd.DataFrame(randGO, columns = go_m.columns,
                     index = go_m.index)
    
    return randGO


def random_dataset_mutation(pat = 500, genes =100, pathway = 50, ratio = 0.5):
    '''
    create a random dataset to test linear pathway
    parameter, number of patients, number of genes, number of pathway
    ratio controlled the mutation dataset
        * number of mutated genes 
    return: 
    X = dataframe with row number (n patients), column (n genes), 
        dummy names as column and index
    go = dataframe with row number (n genes), column (n pathway), 
        dummy names as column and index. first column is genes
    y_enc = y categorically encoded,number of patients
    mut = exome like dataset, the 0 represents no mutation
        1 mutation, return a sample of comumns of patients (genes)
    EXAMPLE USAGE
    X, y, go, mut = random_dataset_mutation(pat = 500, genes =100, pathway = 50, ratio = 0.5)
    '''
    _X = np.random.rand(pat,genes)
    _y = []
    for i in range(0,pat):
        n = random.randint(0,1)
        _y.append(n)
    _y_enc = to_categorical(_y)
    _go =np.random.randint(2, size=(genes, pathway))
    _go = pd.DataFrame(_go, columns = ["go" +str(i) for i in range(1,_go.shape[1] +1,1)],
                     index = ["bla" +str(i) for i in range(1,_go.shape[0] +1,1)])
    _go = _go.reset_index(level=0)
    _go = _go.rename(columns={'index':'genes'})
    _X = pd.DataFrame(_X, columns = ["bla" +str(i) for i in range(1,_X.shape[1] +1,1)],
                     index = ["pat" +str(i) for i in range(1,_X.shape[0] +1,1)])
    ratio = int(genes * ratio)
    mut_col = random.sample(_X.columns.tolist(), ratio)
    _mut = np.random.randint(2, size=(pat, ratio))
    _mut = pd.DataFrame(_mut, columns = mut_col,
                 index = ["pat" +str(i) for i in range(1,_X.shape[0] +1,1)])
    _col = [i +"*" for i in _X.columns.tolist()]
    _X.columns = _col
    
    return _X, _y_enc, _go, _mut


def toy_dataset(x_samples =200, x_featur = 3006, pathways = 1720):
    '''
    create a toy dataset for a linearGO model
    select number of x examples, feature and number of pathways. 
    this is just a toy dataset to test the linearGO model
    EXAMPLE USAGE
    X, y_enc, go = toy_dataset(x_samples =200, x_featur = 3006, pathways = 1720)
    
    '''
    _X = np.random.rand(x_samples,x_featur)
    _y = []
    for i in range(0,x_samples):
        n = random.randint(0,1)
        _y.append(n)
    _y_enc = to_categorical(y)
    _go =np.random.randint(2, size=(x_featur, pathways))
    return _X, _y_enc, _go
    

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

def make_class_dataset(_n_samples=1000, n_feat=100,n_inf=10,n_red=0, n_rep=0, n_clas =6,
        class_sepr=0.8,seed=42):
    '''
    create a dataset to check the importance of the feature. It is creating a dataset where the
    ground truth is known 
    it takes as parameters the make_classification parameters (the scikit-learn function)
    and then it returns X, y dataset (X a pandas dataframe, y array of classes)
    c  is the feature importance (obtained using logistic regression and adding a bias weight)
    Usage example:
    X, y, c = make_class_dataset()
    with redundant:
    X, y, c = make_class_dataset(n_red=10)
    
    '''
    X, y = make_classification(
        n_samples=_n_samples,
        n_features=n_feat,
        n_informative=n_inf,
        n_redundant=n_red,
        n_repeated=n_rep,
        n_classes=n_clas,
        n_clusters_per_class=1,
        class_sep=class_sepr,
        random_state=seed,
        shuffle = False,
    )
    
    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    
    col_names =['col_' + str(i) for i in range(n_feat) ]
    X = pd.DataFrame(X, columns= col_names)
    clf = LogisticRegression(random_state=0).fit(X.iloc[:,:n_inf], y)
    c_inf = np.sum(np.abs(clf.coef_), axis=0)
    c_inf = NormalizeData(c_inf) +10
    
    if n_red is not 0:
        
        clf = LogisticRegression(random_state=0).fit(X.iloc[:,n_inf:(n_inf+n_red)], y)
        c_red = np.sum(np.abs(clf.coef_), axis=0)
        c_red = NormalizeData(c_red) + 5
        clf = LogisticRegression(random_state=0).fit(X.iloc[:,(n_inf+n_red):], y)
        c_noise = np.sum(np.abs(clf.coef_), axis=0)
        c_noise = NormalizeData(c_noise)
        c =list(c_inf) + list(c_red) + list(c_noise)
        
    else:
        
        clf = LogisticRegression(random_state=0).fit(X.iloc[:,n_inf:], y)
        c_noise = np.sum(np.abs(clf.coef_), axis=0)
        c_noise = NormalizeData(c_noise)
        c =list(c_inf) + list(c_noise)
        
    return X, y, c
        
 def constrain_dataset(_n_samples=1000, n_feat=100,n_inf=10,n_red=0, n_rep=0, n_clas =6,
        class_sepr=0.8,seed=42, criterion = 'type_1', pathways= 20, pathway_inf = 0.5, 
                     pathway_red = 0.3):
    '''
    check parameters from make_class_dataset
    this is the extension to use with contrain nets
    criterion: control the connection between the features
        type_1: each type of features is connected only by themselves (ex: informative features
                are connected only with other informative features, but not with redundant ot
                uninformative) -- Default
        type_2: informative and redundant are connected between themselves, but not with 
                uninformative
        type_3: random connections
    pathways = control the number of group (neurons in the the next layer)
    pathway_inf = percentage of pathways connected to informative features
    pathway_red = percentage of pathways connected to redundant features
    return:
    create a dataset to check the importance of the feature.
    it takes as parameters the make_classification parameters (the scikit-learn function)
    and then it returns X, y dataset (X a pandas dataframe, y array of classes)
    c  is the feature importance (obtained using logistic regression and adding a bias weight)
    go a matrix controlling the interaction between features
    Usage example
        X, y, c, go = constrain_dataset()
    
    '''
    
    X, y, c = make_class_dataset(_n_samples=_n_samples, n_feat=n_feat,n_inf=n_inf,n_red=n_red, 
                       n_rep=n_rep, n_clas =n_clas,class_sepr=class_sepr,seed=seed)
    
    if criterion == 'type_1':
        
        if n_red is not 0:
            pathway_inf = int(pathways * pathway_inf)
            pathway_red = int(pathways * pathway_red)
            _go =np.random.randint(1, size=(n_feat, pathways))
            _go[:n_inf, :pathway_inf] = np.random.randint(2, size=(n_inf, pathway_inf))
            _go[n_inf:(n_inf+n_red), pathway_inf:(pathway_inf+ pathway_red)] = \
            np.random.randint(2, size=(n_red, pathway_red))
            _go[(n_inf+n_red):, (pathway_inf+ pathway_red):] = \
            np.random.randint(2, size=((n_feat-(n_inf+ n_red) ), 
                                       (pathways-(pathway_inf + pathway_red) ) ))
            
        else:
            pathway_inf = int(pathways * pathway_inf)

            _go =np.random.randint(1, size=(n_feat, pathways))
            _go[:n_inf, :pathway_inf] = np.random.randint(2, size=(n_inf, pathway_inf))
            _go[n_inf:, pathway_inf:] = np.random.randint(2, size=((n_feat-n_inf), 
                                                                   (pathways-pathway_inf) ))
        
    if criterion == 'type_2':
            pathway_inf = int(pathways * pathway_inf)
            pathway_red = int(pathways * pathway_red)
            path_inf_red =pathway_inf + pathway_red
            n_inf_red = n_inf +n_red
            _go =np.random.randint(1, size=(n_feat, pathways))
            _go[:n_inf_red, :path_inf_red] = np.random.randint(2, size=(n_inf_red, path_inf_red))
            _go[n_inf_red:, path_inf_red:] = np.random.randint(2, size=((n_feat-n_inf_red), 
                                                                   (pathways-path_inf_red) ))
            
    if criterion == 'type_3':
            _go =np.random.randint(2, size=(n_feat, pathways))
        
    
    return X, y, c, _go
      


