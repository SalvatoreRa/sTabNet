import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
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
    
    Bias= True, default, if you want Bias weight addition

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
        
       
            
        
        x = x * alpha

        return x
    



