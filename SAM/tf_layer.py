import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import activations, constraints, regularizers
import tensorflow.keras as keras

"""
LINEAR GO - version 1
linearGO is a dense layer which requires a matrix x and y
INPUT
x: a m*n matrix as input (where m is the number of examples and n the number of features)
y: n x o (a sparse matrix, or a matrix that can have also some weights, n is the number of
the features in x, o the number of the neurons)
FUNCTION
generate a sparse tensor, the weight are the gene-pathway connection
multipy y with the weight matrix
OUTPUT
return a tensor m*o
PARAMETER IMPLEMENTED
* Activation, an activation function from keras activation (default none)
EXAMPLE USAGE1
X, y, go = random_dataset(pat = 500, genes =100, pathway = 50)
linear_layer = LinearGO( zeroes = go, kernel_regularizer='l1')
y = linear_layer(X)
print(y)
EXAMPLE USAGE2
X, y, go, mut = random_dataset_mutation(pat = 10, genes =10, pathway = 5, ratio = 0.5)
concat_test, adj_test, _ = concat_go_matrix(expr = X, exo_df = mut, gene_pats =go, filt_mats = None,
                                            min_max_genes = [1,100], pat_nums = 1)
linear_layer = LinearGO( zeroes = adj_test, kernel_regularizer='l1')
y = linear_layer(concat_test)
print(y)

"""


class LinearGO(keras.layers.Layer):
    def __init__(self, units=3, input_dim=3, zeroes = None,
                 activation=None, kernel_regularizer=None,
                 bias_regularizer=None, **kwargs):
        super(LinearGO, self).__init__( **kwargs)
        self.units = units
        self.zeroes = zeroes
        self.unit_number = int(np.sum(np.array(self.zeroes)))
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activation = activation
        self.activation_fn = activations.get(activation)
        

    def build(self, input_shape):
        self.sparse_mat = tf.convert_to_tensor(self.zeroes , dtype=tf.float32)
        
        self.kernel = self.add_weight(
            shape=(self.unit_number, ),
            initializer="glorot_uniform",
            regularizer=self.kernel_regularizer,
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.sparse_mat.shape[-1],), 
            initializer="glorot_uniform", 
            regularizer=self.bias_regularizer,
            trainable=True
        )
        
        #self.w  = tf.multiply(self.w, self.sparse_mat)
        
    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs , dtype=tf.float32)
        self.idx_sparse = tf.where(tf.not_equal(self.sparse_mat, 0))
        self.sparse = tf.SparseTensor(self.idx_sparse, self.kernel, 
                                      self.sparse_mat.get_shape())          
       
        output = tf.sparse.sparse_dense_matmul(inputs, self.sparse ) + self.b 
        if self.activation_fn is not None:
            output = self.activation_fn(output)
        
        return output
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.unit_number)
    
    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({"units": self.units})
        return config

import tensorflow as tf
import pandas as pd
import seaborn as sns

class Linear_data():
    '''
    data storage for the weights
    plotting is implemented
    a = Linear_data( layer = linear_layer, path_mat = go)
    '''
    def __init__ (self, layer = None, path_mat= None  ):
        self._layer = layer
        self._path_mat = path_mat
        self.crude_weight, self.crude_bias = self.weight_rescue()
        self.annot_bias, self.annot_weight = self.weight_annot()
        self.w_mean = self.weight_means()
    
    def weight_rescue(self):
        _w = self._layer.weights[0].numpy()
        _b = self._layer.weights[1].numpy()
        return _w, _b
    
    def weight_annot(self):
        _w = self._layer.weights[0].numpy()
        _b = self._layer.weights[1].numpy()
        
        _b = pd.Series(_b, index = self._path_mat.columns)
        sparse_mat = tf.convert_to_tensor(self._path_mat, dtype=tf.float32)
        idx_sparse = tf.where(tf.not_equal(sparse_mat, 0))
        sparse = tf.SparseTensor(idx_sparse, _w, sparse_mat.get_shape())
        _wm = tf.sparse.to_dense(sparse).numpy()
        _wm = pd.DataFrame(_wm, index = self._path_mat.index,  
                           columns = self._path_mat.columns).T
        _wm = pd.melt(_wm.reset_index(), id_vars='index')
        _wm = _wm[_wm.value != 0]
        _wm.columns =  ["to_node", "from_node", "weight_value"]
        return _b, _wm
    
    def weight_means(self):
        _w = self.annot_weight.sort_values(by=['weight_value'], ascending=False)
        df =  _w .groupby(['from_node']).agg(['mean', 'count'])
        df.columns = [ '_'.join(str(i) for i in col) for col in df.columns]
        df.reset_index( inplace=True)
        df =df.reindex(df.weight_value_mean.abs().sort_values(ascending=False).index)
        return df
    
    def weight_means_pat(self):
        _w = self.annot_weight.sort_values(by=['weight_value'], ascending=False)
        df =  _w .groupby(['to_node']).agg(['mean', 'count'])
        df.columns = [ '_'.join(str(i) for i in col) for col in df.columns]
        df.reset_index( inplace=True)
        df =df.reindex(df.weight_value_mean.abs().sort_values(ascending=False).index)
        return df
    
    def plot_weights(self, n = 10):
        weights = self.annot_weight.sort_values(by=['weight_value'], 
                                                ascending=False).iloc[0:n,:]
        weights["Input"] = [str(i) for i in range(n)]
        fig, ax = plt.subplots()    
        ax = sns.barplot(y = "Input", x= "weight_value", data = weights)
        ax.set_yticklabels(weights['from_node'])  

        k =weights["to_node"].to_list()
        for bar, i in zip(ax.patches, k[::-1]):
            ax.text(0, bar.get_y()+bar.get_height()/2,  i, color = 'black', 
                    ha = 'left', va = 'center')
    
    def weight_means_plot(self, n =10):
        _w = self.annot_weight.sort_values(by=['weight_value'], ascending=False)
        df =  _w .groupby(['from_node']).agg(['mean', 'count'])
        df.columns = [ '_'.join(str(i) for i in col) for col in df.columns]
        df.reset_index( inplace=True)
        df =df.reindex(df.weight_value_mean.abs().sort_values(ascending=False).index)
        df = df.iloc[0:n,:].sort_values(by=['weight_value_mean'], ascending=False)
        sns.barplot(data = df, x = 'weight_value_mean', y =  'from_node')
        
class Linear_dataloop():
    '''
    Version 1.1
    data storage for the weights in a loop
    stores all the weights from the different layers
    a = Linear_dataloop( layer_gene = linear_layer, path_mat = go)
    '''
    def __init__ (self, layer_gene = None, path_mat= None, w_norm = None,
                 layer_concat = None, conc_mat = None,
                 layer_path = None):
        self._layer = layer_gene
        self._path_mat = path_mat
        self.w_norm = w_norm
        self._layer_concat = layer_concat
        self._conc_mat = conc_mat
        self._layer_path = layer_path
    
    def weight_annot(self):
        ''' annoting the weights'''
        _w = self._layer.weights[0].numpy()
        _b = self._layer.weights[1].numpy()
        
        _b = pd.Series(_b, index = self._path_mat.columns)
        sparse_mat = tf.convert_to_tensor(self._path_mat, dtype=tf.float32)
        idx_sparse = tf.where(tf.not_equal(sparse_mat, 0))
        sparse = tf.SparseTensor(idx_sparse, _w, sparse_mat.get_shape())
        _wm = tf.sparse.to_dense(sparse).numpy()
        _wm = pd.DataFrame(_wm, index = self._path_mat.index,  
                           columns = self._path_mat.columns).T
        _wm = pd.melt(_wm.reset_index(), id_vars='index')
        _wm = _wm[_wm.value != 0]
        _wm.columns =  ["to_node", "from_node", "weight_value"]
        return _b, _wm
    
    def weight_means(self):
        ''' average the weights (from gene)
            normalization: if the input node has more than 5 sigma connections
            divided by the number of connections
        '''
        _, _w = self.weight_annot()
        _w = _w.sort_values(by=['weight_value'], ascending=False)
        _df =  _w .groupby(['from_node']).agg(['mean', 'count'])
        _df.columns = [ '_'.join(str(i) for i in col) for col in _df.columns]
        _df.reset_index( inplace=True)
        _df =_df.reindex(_df.weight_value_mean.abs().sort_values(ascending=False).index)
        if self.w_norm == "sigma_5":
            _w = _w.sort_values(by=['weight_value'], ascending=False)
            _df =  _w .groupby(['from_node']).agg(['sum', 'count'])
            _df.columns = [ '_'.join(str(i) for i in col) for col in _df.columns]
            _df.reset_index( inplace=True)
            _df =_df.reindex(_df.weight_value_sum.abs().sort_values(ascending=False).index)
            sigma_5 = np.std(_df.to_node_count) * 5
            _df["weight_value_mean"] =np.where(_df.iloc[:,2] >= sigma_5, _df.iloc[:,3]/_df.iloc[:,2],
                                             _df.iloc[:,3])
            _df = _df[["from_node", "weight_value_mean", "weight_value_count"]]
        return _df
    
    def weight_p(self):
        ''' average the weights (to gene)
            normalization: if the input node has more than 5 sigma connections
            divided by the number of connections
        '''
        _, _w = self.weight_annot()
        _w = _w.sort_values(by=['weight_value'], ascending=False)
        _df =  _w .groupby(['to_node']).agg(['mean', 'count'])
        _df.columns = [ '_'.join(str(i) for i in col) for col in _df.columns]
        _df.reset_index( inplace=True)
        _df =_df.reindex(_df.weight_value_mean.abs().sort_values(ascending=False).index)
        if self.w_norm == "sigma_5":
            _w = _w.sort_values(by=['weight_value'], ascending=False)
            _df =  _w .groupby(['to_node']).agg(['sum', 'count'])
            _df.columns = [ '_'.join(str(i) for i in col) for col in _df.columns]
            _df.reset_index( inplace=True)
            _df =_df.reindex(_df.weight_value_sum.abs().sort_values(ascending=False).index)
            sigma_5 = np.std(_df.from_node_count) * 5
            _df["weight_value_mean"] =np.where(_df.iloc[:,2] >= sigma_5, _df.iloc[:,3]/_df.iloc[:,2],
                                                 _df.iloc[:,3])
            _df = _df[["to_node", "weight_value_mean", "weight_value_count"]]

        return _df
    
    def weight_annot_conc(self):
        ''' annoting the weights for the concat layer'''
        _w = self._layer_concat.weights[0].numpy()
        _b = self._layer_concat.weights[1].numpy()
        
        _b = pd.Series(_b, index = self._conc_mat.columns)
        sparse_mat = tf.convert_to_tensor(self._conc_mat, dtype=tf.float32)
        idx_sparse = tf.where(tf.not_equal(sparse_mat, 0))
        sparse = tf.SparseTensor(idx_sparse, _w, sparse_mat.get_shape())
        _wm = tf.sparse.to_dense(sparse).numpy()
        _wm = pd.DataFrame(_wm, index = self._conc_mat.index,  
                           columns = self._conc_mat.columns).T
        _wm = pd.melt(_wm.reset_index(), id_vars='index')
        _wm = _wm[_wm.value != 0]
        _wm.columns =  ["to_node", "from_node", "weight_value"]
        return _b, _wm
    
    def weight_means_concat(self):
        ''' return the weights for gene and mutation
        '''
        _, _w = self.weight_annot_conc()
        _w = _w.sort_values(by=['weight_value'], ascending=False)
        _df =  _w .groupby(['from_node']).agg(['mean', 'count'])
        _df.columns = [ '_'.join(str(i) for i in col) for col in _df.columns]
        _df.reset_index( inplace=True)
        _df =_df.reindex(_df.weight_value_mean.abs().sort_values(ascending=False).index)
        return _df
    
    def weight_annot_path(self):
        ''' annoting the weights for path layer'''
        _w = self._layer_path.weights[0].numpy()
        _b = self._layer_path.weights[1].numpy()
        return _b , _w
    
    def weight_means_path(self):
        ''' annoting the weights average for path layer'''
        _b, _wm = self.weight_annot_path()
        _wm =pd.DataFrame(_wm, index = self._path_mat.columns)

        _wm["avg"] = _wm.iloc[:,0:_wm.shape[1]].mean(axis= 1)
        _wm =_wm.reindex(_wm.avg.abs().sort_values(ascending=False).index)

        _wm["pathway"] = _wm.index
        _wm = _wm[["pathway", "avg"]]
        return _wm

    def loop_go(self):
        ''' easy returning of data for loop '''
        _df = self.weight_means().iloc[:,0:2]
        _df1 = self.weight_p().iloc[:,0:2]
        
        return _df, _df1
    
    def loop_go_conc(self):
        ''' easy returning of data for loop for concat'''
        _df = self.weight_means().iloc[:,0:2] #gene value
        _df1 = self.weight_p().iloc[:,0:2] #pathway from gene
        _df2 = self.weight_means_concat().iloc[:,0:2] #block component
        _df3 = self.weight_means_path() #pathway from layer
        
        return _df, _df1, _df2, _df3
    
    def loop_go_conc1(self):
        ''' easy returning of data for loop for concat
            gene value   
        '''
        _df = self.weight_means().iloc[:,0:2] #gene value
        return _df
    
    def loop_go_conc2(self):
        ''' easy returning of data for loop for concat
            #pathway from gene 
        '''
        _df = self.weight_p().iloc[:,0:2] #pathway from gene
        return _df
    
    def loop_go_conc3(self):
        ''' easy returning of data for loop for concat
            #block component
            weight from mutation and gene level
        '''
        _df = self.weight_means_concat().iloc[:,0:2] #block component
        return _df
    
    def loop_go_conc4(self):
        ''' easy returning of data for loop for concat
            #pathway from layer 
        '''
        _df = self.weight_means_path() #pathway from layer
        return _df
    

class Linear_dataloop1():
    '''
    version 1.2
    data storage for the weights in a loop
    stores all the weights from the different layers
    a = Linear_dataloop( layer_gene = linear_layer, path_mat = go)
    '''
    def __init__ (self, layer_gene = None, path_mat= None, w_norm = None,
                 layer_concat = None, conc_mat = None,
                 layer_path = None):
        self._layer = layer_gene
        self._path_mat = path_mat
        self.w_norm = w_norm
        self._layer_concat = layer_concat
        self._conc_mat = conc_mat
        self._layer_path = layer_path
    
    def weight_annot(self):
        ''' annoting the weights'''
        _w = self._layer.weights[0].numpy()
        _b = self._layer.weights[1].numpy()
        
        _b = pd.Series(_b, index = self._path_mat.columns)
        sparse_mat = tf.convert_to_tensor(self._path_mat, dtype=tf.float32)
        idx_sparse = tf.where(tf.not_equal(sparse_mat, 0))
        sparse = tf.SparseTensor(idx_sparse, _w, sparse_mat.get_shape())
        _wm = tf.sparse.to_dense(sparse).numpy()
        _wm = pd.DataFrame(_wm, index = self._path_mat.index,  
                           columns = self._path_mat.columns).T
        _wm = pd.melt(_wm.reset_index(), id_vars='index')
        _wm = _wm[_wm.value != 0]
        _wm.columns =  ["to_node", "from_node", "weight_value"]
        return _b, _wm
    
    def weight_means(self):
        ''' average the weights (from gene)
            normalization: if the input node has more than 5 sigma connections
            divided by the number of connections
        '''
        _, _w = self.weight_annot()
        _w = _w.sort_values(by=['weight_value'], ascending=False)
        _df =  _w .groupby(['from_node']).agg(['mean', 'count'])
        _df.columns = [ '_'.join(str(i) for i in col) for col in _df.columns]
        _df.reset_index( inplace=True)
        _df =_df.reindex(_df.weight_value_mean.abs().sort_values(ascending=False).index)
        if self.w_norm == "sigma_5":
            _w = _w.sort_values(by=['weight_value'], ascending=False)
            _df =  _w .groupby(['from_node']).agg(['sum', 'count'])
            _df.columns = [ '_'.join(str(i) for i in col) for col in _df.columns]
            _df.reset_index( inplace=True)
            _df =_df.reindex(_df.weight_value_sum.abs().sort_values(ascending=False).index)
            sigma_5 = np.std(_df.to_node_count) * 5
            _df["weight_value_mean"] =np.where(_df.iloc[:,2] >= sigma_5, _df.iloc[:,3]/_df.iloc[:,2],
                                             _df.iloc[:,3])
            _df = _df[["from_node", "weight_value_mean", "weight_value_count"]]
        return _df
    
    def weight_p(self):
        ''' average the weights (to gene)
            normalization: if the input node has more than 5 sigma connections
            divided by the number of connections
        '''
        _, _w = self.weight_annot()
        _w = _w.sort_values(by=['weight_value'], ascending=False)
        _df =  _w .groupby(['to_node']).agg(['mean', 'count'])
        _df.columns = [ '_'.join(str(i) for i in col) for col in _df.columns]
        _df.reset_index( inplace=True)
        _df =_df.reindex(_df.weight_value_mean.abs().sort_values(ascending=False).index)
        if self.w_norm == "sigma_5":
            _w = _w.sort_values(by=['weight_value'], ascending=False)
            _df =  _w .groupby(['to_node']).agg(['sum', 'count'])
            _df.columns = [ '_'.join(str(i) for i in col) for col in _df.columns]
            _df.reset_index( inplace=True)
            _df =_df.reindex(_df.weight_value_sum.abs().sort_values(ascending=False).index)
            sigma_5 = np.std(_df.from_node_count) * 5
            _df["weight_value_mean"] =np.where(_df.iloc[:,2] >= sigma_5, _df.iloc[:,3]/_df.iloc[:,2],
                                                 _df.iloc[:,3])
            _df = _df[["to_node", "weight_value_mean", "weight_value_count"]]

        return _df
    
    def weight_annot_conc(self):
        ''' annoting the weights for the concat layer'''
        _w = self._layer_concat.weights[0].numpy()
        _b = self._layer_concat.weights[1].numpy()
        
        _b = pd.Series(_b, index = self._conc_mat.columns)
        sparse_mat = tf.convert_to_tensor(self._conc_mat, dtype=tf.float32)
        idx_sparse = tf.where(tf.not_equal(sparse_mat, 0))
        sparse = tf.SparseTensor(idx_sparse, _w, sparse_mat.get_shape())
        _wm = tf.sparse.to_dense(sparse).numpy()
        _wm = pd.DataFrame(_wm, index = self._conc_mat.index,  
                           columns = self._conc_mat.columns).T
        _wm = pd.melt(_wm.reset_index(), id_vars='index')
        _wm = _wm[_wm.value != 0]
        _wm.columns =  ["to_node", "from_node", "weight_value"]
        return _b, _wm
    
    def weight_means_concat(self):
        ''' return the weights for gene and mutation
        '''
        _, _w = self.weight_annot_conc()
        _w = _w.sort_values(by=['weight_value'], ascending=False)
        _df =  _w .groupby(['from_node']).agg(['mean', 'count'])
        _df.columns = [ '_'.join(str(i) for i in col) for col in _df.columns]
        _df.reset_index( inplace=True)
        _df =_df.reindex(_df.weight_value_mean.abs().sort_values(ascending=False).index)
        return _df
    
    def weight_annot_path(self):
        ''' annoting the weights for path layer'''
        _w = self._layer_path.weights[0].numpy()
        _b = self._layer_path.weights[1].numpy()
        return _b , _w
    
    def weight_means_path(self):
        ''' annoting the weights average for path layer'''
        _b, _wm = self.weight_annot_path()
        _wm =pd.DataFrame(_wm, index = self._path_mat.columns)

        _wm["avg"] = _wm.iloc[:,0:_wm.shape[1]].mean(axis= 1)
        _wm =_wm.reindex(_wm.avg.abs().sort_values(ascending=False).index)

        _wm["pathway"] = _wm.index
        _wm = _wm[["pathway", "avg"]]
        return _wm

    def loop_go(self):
        ''' easy returning of data for loop '''
        _df = self.weight_means().iloc[:,0:2]
        _df1 = self.weight_p().iloc[:,0:2]
        
        return _df, _df1
    
    def loop_go_conc(self):
        ''' easy returning of data for loop for concat'''
        _df = self.weight_means().iloc[:,0:2] #gene value
        _df1 = self.weight_p().iloc[:,0:2] #pathway from gene
        _df2 = self.weight_means_concat().iloc[:,0:2] #block component
        _df3 = self.weight_means_path() #pathway from layer
        
        return _df, _df1, _df2, _df3
    
    def loop_go_conc1(self):
        ''' easy returning of data for loop for concat
            gene value   
        '''
        _df = self.weight_means().iloc[:,0:2] #gene value
        return _df
    
    def loop_go_conc2(self):
        ''' easy returning of data for loop for concat
            #pathway from gene 
        '''
        _df = self.weight_p().iloc[:,0:2] #pathway from gene
        return _df
    
    def loop_go_conc3(self):
        ''' easy returning of data for loop for concat
            #block component
            weight from mutation and gene level
        '''
        _df = self.weight_means_concat().iloc[:,0:2] #block component
        return _df
    
    def loop_go_conc4(self):
        ''' easy returning of data for loop for concat
            #pathway from layer 
        '''
        _df = self.weight_means_path() #pathway from layer
        return _df
        
    
    
# Add attention layer to the deep learning network
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
 
    def build(self,input_shape):
        
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='glorot_uniform', trainable=True)
        if self.bias:
            self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],), 
                               initializer='glorot_uniform', trainable=True)        
        super(attention, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        
        if self.mechanism=="Bahdanau":        
            if self.bias:
                e = tf.tanh(tf.matmul(x,self.W +self.b))
            else:
                e = tf.tanh(tf.matmul(x,self.W ))
            alpha = tf.nn.softmax(e)
            
        if self.mechanism=="Luong":        
            if self.bias:
                e = tf.matmul(x,self.W +self.b)
            else:
                e = tf.matmul(x,self.W )
            alpha = tf.nn.softmax(e)
            
        if self.mechanism=="Graves":        
            if self.bias:
                e = tf.math.cos(tf.matmul(x,self.W +self.b))
            else:
                e = tf.math.cos(tf.matmul(x,self.W ))
            alpha = tf.nn.softmax(e)
            
        if self.mechanism=="scaled_dot":        
            if self.bias:
                e = tf.matmul(x,self.W +self.b)
            else:
                e = tf.matmul(x,self.W )
            
            scaling_factor = tf.math.rsqrt(tf.convert_to_tensor((x.shape[-1]),
                                                                dtype=tf.float32 ))
            e = tf.multiply(e,scaling_factor )
            alpha = tf.nn.softmax(e)
        
        if self.mechanism=="exp1":        
            if self.bias:
                e = tf.matmul(x,self.W +self.b)
            else:
                e = tf.matmul(x,self.W )
            
            scaling_factor = tf.math.rsqrt(tf.convert_to_tensor((x.shape[-1]),
                                                                dtype=tf.float32 ))
            e = tf.keras.activations.swish(e)
            e = tf.multiply(e,scaling_factor )
            alpha = tf.nn.softmax(e)
            
        
        x = x * alpha

        return x
    

