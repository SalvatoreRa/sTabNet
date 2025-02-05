import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split, SubsetRandomSampler, ConcatDataset, Dataset

class AttentionLinear(nn.Linear):
    
    """
    defines attention layer
    """
    
    def __init__(self, in_features, out_features,  bias= True, mechanism="Bahdanau"):
        super(AttentionLinear,self).__init__(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.nn.init.xavier_uniform_(nn.Parameter(torch.empty((in_features,1))), gain=1.0))
        
        if bias == True:
            self.bias = nn.Parameter(nn.init.xavier_uniform_(nn.Parameter(torch.empty(in_features))))
        else:
            self.register_parameter('bias', None)
            
        self.mechanism = mechanism
        self.alpha = nn.Parameter(nn.init.normal_(nn.Parameter(torch.empty(in_features))))
        
        

        
    def forward(self,x):
    # Alignment scores. Pass them through tanh function
        
        alpha = self.alpha
    
        if self.mechanism=="Bahdanau":        
            if self.bias == True:
                e = nn.Tanh(torch.matmul(x,self.weight +self.bias))
            else:
                e = nn.Tanh(torch.matmul(x,self.weight ))
            alpha = nn.Softmax(dim=1)(e) * self.alpha
            
        if self.mechanism=="Luong":        
            if self.bias == True:
                e = torch.matmul(x,self.weight +self.bias)
            else:
                e = torch.matmul(x,self.weight )
            alpha = nn.Softmax(dim=1)(e) * self.alpha
            
        if self.mechanism=="Graves":        
            if self.bias == True:
                e = torch.cos(torch.matmul(x,self.weight +self.bias))
            else:
                e = torch.cos(torch.matmul(x,self.weight ))
            alpha = nn.Softmax(dim=1)(e) * self.alpha
            
        if self.mechanism=="scaled_dot":        
            if self.bias == True:
                e = torch.matmul(x,self.weight +self.bias)
            else:
                e = torch.matmul(x,self.weight ) 
            
            scaling_factor = torch.rsqrt(torch.tensor(x.shape[-1]))
            e = torch.mul(e,scaling_factor )
            alpha = nn.Softmax(dim=1)(e)* self.alpha
        

        return x