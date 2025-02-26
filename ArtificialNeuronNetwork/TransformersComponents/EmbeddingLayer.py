'''
Created on 25 févr. 2025

@author: SSM9
'''

from ArtificialNeuronNetwork.NeuronNetwork import NeuronNetwork
from ArtificialNeuronNetwork import Activation_functions
from ArtificialNeuronNetwork import Cost_functions
from ArtificialNeuronNetwork.Neuron import Optimizer

from enum import Enum

import numpy as np

class EmbeddingLayer(object):
    '''
    Embedding layer is just a linear layer having as input size the vocabulary size which is one hot encoded and  as output size the embedding depth
    '''
    
    vocabulary_size = None
    embedding_depth = None
    
    embedding_layer = None
    weights = None


    def __init__(self, vocabulary_size=0, embedding_depth=256):
        
        self.vocabulary_size = vocabulary_size
        self.embedding_depth = embedding_depth
        
        #CODE : usage of Neuron network class - poor performance compared to simple lookup table
        #self.embedding_layer = NeuronNetwork(number_of_inputs= self.vocabulary_size, number_of_outputs= self.embedding_depth, correction_coeff = 1e-2,optimizer = Optimizer.ADAM,beta1 = 0.9,beta2 = 0.999)
        
        #CODE : replace call to neurons to simple matrix instance of shape(self.vocabulary_size,self.embedding_depth)
        self.weights = np.random.rand(self.vocabulary_size,self.embedding_depth)
        
    def __call__(self, input_token_ids):
        
        #CODE : replace execute Neuron network class method by simple lookup table for better performance
        if input_token_ids.ndim == 1 :        
            my_embedded_token_IDs = np.array([self.weights[token_id] for token_id in input_token_ids])
        else :
            my_embedded_token_IDs=[]
            for i in range (0, len(input_token_ids),1):
                my_embedded_token_IDs.append(np.array([self.weights[token_id] for token_id in input_token_ids[i]]))
        
        '''
        #Code using neuron network class  
                
        if input_token_ids.ndim == 1 :        
            my_one_encoded_inputs = np.array([self.one_hot_encoding(token_id) for token_id in input_token_ids])
            my_embedded_token_IDs = self.embedding_layer.executeModelOnBatch(my_one_encoded_inputs)
        else :
            my_embedded_token_IDs=np.zeros((input_token_ids.shape[0],input_token_ids.shape[1],self.embedding_depth))
            for i in range (0, len(input_token_ids),1):
                my_one_encoded_inputs = np.array([self.one_hot_encoding(token_id) for token_id in input_token_ids[i]])
                my_embedded_token_IDs[i]= self.embedding_layer.executeModelOnBatch(my_one_encoded_inputs)
        '''
        return np.array(my_embedded_token_IDs)
    
       
    def one_hot_encoding(self, token_id):
    
        one_encoded_token_id = np.zeros(self.vocabulary_size)
        one_encoded_token_id[token_id]=1
        return one_encoded_token_id
    
    #TODO back propagation   
    
    
class PositionEmbeddingLayer(object):
    '''
    Position Embedding layer is just a linear layer having as input size the vocabulary size which is one hot encoded and  as output size the embedding depth
    '''
    context_size = None
    embedding_depth = None
    
    pos_embedding_layer = None
    
    pos_embedding_type=None
    
    def __init__(self, context_size=256, embedding_depth=256, pos_embedding_type=None):
        
        if pos_embedding_type == None:
            self.pos_embedding_type = Pos_embedding_type.ABSOLUTE
        else:
            self.pos_embedding_type = pos_embedding_type
        
        self.context_size = context_size
        self.embedding_depth = embedding_depth
        
        self.pos_embedding_layer = NeuronNetwork(number_of_inputs= 1, number_of_outputs= self.embedding_depth, correction_coeff = 1e-2,optimizer = Optimizer.ADAM,beta1 = 0.9,beta2 = 0.999)
        
        
    def __call__(self, embeddedInputs):
        
        if(self.pos_embedding_type == Pos_embedding_type.ABSOLUTE):
            return self.absolutePosEmbedding(embeddedInputs)
        else:
            return self.absolutePosEmbedding(embeddedInputs)
            
    
    def absolutePosEmbedding(self, embeddedInputs):
        #Je crée mon vecteur de positions absolues "embeddées"
        my_pos_embdeddings =  np.resize(self.pos_embedding_layer.executeModelOnBatch(np.resize(np.arange(0,self.context_size,1),(self.context_size,1))),(self.context_size,self.embedding_depth))
        #Je renvoie mes données d'entrée "embeddées" additionnées avec les infos de positions
        return np.add(embeddedInputs,my_pos_embdeddings)
    

class Pos_embedding_type(Enum):
    RELATIVE = 0,
    ABSOLUTE = 1