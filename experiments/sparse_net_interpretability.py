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

'''
we are importing some function to make the code quicker
linearGO is a defined layer that is enforcing sparsity,
classical linear layer: XW + B, where W is a matrix and B the bias
LinearGO: X (W @ A) + B, where A is a binary matrix (0/1) controlling
	the connections, thus there is a weight between a neuron only if there is
	a definition in A. @ is the hadamard product. the layer is implemented using
	sparse matrix multiplication to be more efficient

'''

import toy_datasets, tf_layer, tools, matrixes
from tools import keras_cat, history_plot, splitting_data
from tf_layer import LinearGO 



##### Attention layer ######

class attention(keras.layers.Layer):
    """
    attention layer v2.1
    derived from the Bahdanau and Luong Attention Mechanism 
    keras custom layer that takes a batch as input.
    The layer is conservative of the dimension of the input (i.e input and output
    have the same dimensions)
    Return an input that is scaled and non linear transformed trough attention weights
    attention weights can be retrieved as usual weights in keras layer
    usage: as classical keras layer (is a subclass of a keras layer)
    it works with classical FFNN, Biological NN etc...
    advice:
    * train for more epochs
    mechanis
    mechanism="Bahdanau", define the Badhanau
    mechanism="Luong", define the luong
    echanism="scaled_dot", define the luong but scaled for number of features
    mechanism ="Graves", defined as Graves
    mechanism ="exp1", dexperimental
    Bias= True, default, if you want Bias weight addition
    in P-net bias are not advised
    """
    def __init__(self, bias= True, mechanism="Bahdanau", **kwargs):
        super(attention,self).__init__(**kwargs)
        self.bias = bias
        self.mechanism = mechanism
        #self.alpha = tf.ones(1, dtype=tf.dtypes.float32)
        
 
    def build(self,input_shape):
        
        self.W=self.add_weight(name='multipl_weight', shape=(input_shape[-1],1), 
                               initializer='glorot_uniform', trainable=True)
        if self.bias:
            self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],), 
                               initializer='glorot_uniform', trainable=True)  
        self.alpha = self.add_weight(name='attention_weight', shape=(input_shape[-1],), 
                               initializer='glorot_uniform', trainable=True)
        super(attention, self).build(input_shape)
        
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        
        if self.mechanism=="Bahdanau":        
            if self.bias:
                e = tf.tanh(tf.matmul(x,self.W +self.b))
            else:
                e = tf.tanh(tf.matmul(x,self.W ))
            alpha = tf.nn.softmax(e) * self.alpha
            
        if self.mechanism=="Luong":        
            if self.bias:
                e = tf.matmul(x,self.W +self.b)
            else:
                e = tf.matmul(x,self.W )
            alpha = tf.nn.softmax(e) * self.alpha
            
        if self.mechanism=="Graves":        
            if self.bias:
                e = tf.math.cos(tf.matmul(x,self.W +self.b))
            else:
                e = tf.math.cos(tf.matmul(x,self.W ))
            alpha = tf.nn.softmax(e) * self.alpha
            
        if self.mechanism=="scaled_dot":        
            if self.bias:
                e = tf.matmul(x,self.W +self.b)
            else:
                e = tf.matmul(x,self.W ) 
            
            scaling_factor = tf.math.rsqrt(tf.convert_to_tensor((x.shape[-1]),
                                                                dtype=tf.float32 ))
            e = tf.multiply(e,scaling_factor )
            alpha = tf.nn.softmax(e) * self.alpha
        
        if self.mechanism=="exp1":        
            if self.bias:
                e = tf.matmul(x,self.W +self.b)
            else:
                e = tf.matmul(x,self.W ) 
            
            scaling_factor = tf.math.rsqrt(tf.convert_to_tensor((x.shape[-1]),
                                                                dtype=tf.float32 ))
            e = tf.keras.activations.swish(e)
            e = tf.multiply(e,scaling_factor )
            alpha  = tf.nn.softmax(e) * self.alpha
            
        
        x = x * alpha

        return x
    
###### dataset generation #######

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

def make_class_dataset(_n_samples=1000, n_feat=100,n_inf=10,n_red=0, n_rep=0, n_clas =6,
        class_sepr=0.8,seed=42):
    '''
    create a dataset to check the importance of the feature.
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
    
    # CREATE DATASET AND FEATURE IMPORTANCE
    class_sep = p
    
    X, y, c, go = constrain_dataset(_n_samples=1000, n_feat=100,n_inf=10,n_red=0, n_rep=0, 
                                    n_clas =6, class_sepr=p,seed=42, criterion = 'type_1', 
                                    pathways= 100, pathway_inf = 0.9, pathway_red = 0.3)
    
    
    for i in range(n):
        X_train, X_test, X_val, y_train_enc, y_val_enc, y_test_enc =splitting_data( data_x = X, 
                                                                            data_y = pd.Series(y), 
                                                                            val_split = 0.1, 
                                                                            test_split = 0.2,
                                                                            random_seed =42)

        inputs = keras.layers.Input(shape =(X_train.shape[-1],))
        x = attention(mechanism="scaled_dot",bias = False)(inputs)
        x = LinearGO(256, zeroes= go, activation ="tanh")(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        x = keras.layers.Dense(64, activation ="relu")(x)
        x = keras.layers.Dropout(rate=0.3)(x)
        x = keras.layers.Dense(y_train_enc.shape[1], activation ="softmax")(x)
        model_go = keras.models.Model(inputs, x)

        model_go.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_go.summary()
        history = model_go.fit(
            X_train,
            y_train_enc,
            batch_size=32,
            epochs=200,
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
        #results.loc[i, "c_ind_red"] =concordance_index(c[n_inf:(n_inf+n_red)],
        #                                               model.feature_importances_[n_inf:(n_inf+n_red)])
        
        attn =model_go.layers[1].weights[1].numpy()
        attn = np.abs(attn).flatten().tolist()
        
        results.loc[i, "c_index"] =concordance_index(c, attn)
        results.loc[i, "c_ind_inf"] =concordance_index(c[:n_inf], attn[:n_inf])
        
        
        results.loc[i, "c_ind_Ninf"] =concordance_index(c[n_inf:], 
                                                       attn[n_inf:])
        results.loc[i, "c_ratio"] =np.sum(attn[:n_inf]) / np.sum(attn[n_inf:])
        feat_imp.loc[:,i] = attn
        
    res_dir = './results/'
        
    results.to_csv(res_dir+ 'results_pnet_inf+' + str(p) +'.csv')
    feat_imp.to_csv(res_dir+ 'feat_imp_pnet_inf+' + str(p) +'.csv')
        
    feat_imp['col_feat'] = feat_imp.index
    feats = feat_imp.melt('col_feat',  var_name='splits', value_name='feat_imp')
    sns_plot =sns.lineplot(data=feats, x="col_feat", y="feat_imp")
    sns_plot.figure.savefig(res_dir+ 'feat_plot_pnet_inf+' + str(p) +'.png')
    plt.close()
