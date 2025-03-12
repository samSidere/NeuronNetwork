'''
Created on 3 déc. 2024

@author: SSM9
'''

import torch
import torch.nn.functional as F


class EmbeddingLayer(object):
    #Embedding => transform data into vector of dimension embeddingDepth
    dictionary = None
    embedding = None
    embeddingDepth = None
    
    output = None
    
    def __init__(self, dc, embeddingDepth=512) :
        
        self.dictionary = dc
        self.embeddingDepth = embeddingDepth
        self.embedding = torch.nn.Embedding(len(self.dictionary), self.embeddingDepth)
        
    def processInputs(self, inputData):
        
        inputData_2_IntSequence = torch.tensor([self.dictionary[inputData] for inputData in sentence.replace(',', '').split()])
        
        embedded_inputData = self.embedding(inputData_2_IntSequence).detach()
        
        self.output = embedded_inputData
        
        return self.output

class MultiHeadedAttentionLayer(object):
    
    h = None
    d = None  

    d_q = None 
    d_k = None 
    d_v = None 
    
    W_query = None
    W_key = None
    W_value = None
    
    output = None
    
    def __init__(self, h=1, d=512, d_q=24, d_k=24, d_v=28) :
        
        self.h = h
        self.d = d  

        self.d_q = d_q 
        self.d_k = d_k 
        self.d_v = d_v
              
        self.W_query = torch.nn.Parameter(torch.rand(self.d_q, self.d))
        self.W_key = torch.nn.Parameter(torch.rand(self.d_k, self.d))
        self.W_value = torch.nn.Parameter(torch.rand(self.d_v, self.d))
    
        self.multihead_W_query = torch.nn.Parameter(torch.rand(self.h, self.d_q, self.d))
        self.multihead_W_key = torch.nn.Parameter(torch.rand(self.h, self.d_k, self.d))
        self.multihead_W_value = torch.nn.Parameter(torch.rand(self.h, self.d_v, self.d))
        
    def processInputs(self, multihead_queries_input, multihead_keys_input, multihead_values_input):
        
        multihead_queries = torch.bmm(multihead_queries_input, self.multihead_W_query.permute(0, 2, 1))
        multihead_keys = torch.bmm(multihead_keys_input, self.multihead_W_key.permute(0, 2, 1))
        multihead_values = torch.bmm(multihead_values_input, self.multihead_W_value.permute(0, 2, 1))
        
        print("multihead_queries.shape:", multihead_queries.shape)
        print("multihead_keys.shape:", multihead_keys.shape)
        print("multihead_values.shape:", multihead_values.shape)
        
        multihead_omegas = torch.bmm(multihead_queries,multihead_keys.permute(0, 2, 1))
        print("multihead_omegas.shape:", multihead_omegas.shape)
        
        multihead_attention_weights = F.softmax(multihead_omegas / self.d_k**0.5, dim=2)
        print("multihead_attention_weights.shape:", multihead_attention_weights.shape)
        #print(multihead_attention_weights)
    
        multihead_context_vectors = torch.bmm(multihead_attention_weights, multihead_values)
        print("multihead_context_vectors.shape:", multihead_context_vectors.shape)
        
        self.output = multihead_context_vectors
        
        return self.output
 
    
    
if __name__ == '__main__':
    
    glossary = "Life is short eat dessert first do have are I you he she we they have has not"
    
    sentence = 'Life is short, eat dessert first'

    dc = {s:i for i,s in enumerate(sorted(glossary.split()))}
    print(dc)
    
    #torch.manual_seed(147)
    
    embeddingLayer = EmbeddingLayer(dc, 512)
    embeddingLayer.processInputs(sentence)
    
    embedded_sentence = embeddingLayer.output

    print(embedded_sentence)
    print(embedded_sentence.shape)
    
    d = embedded_sentence.shape[1]
    
    #nombre de tête (h) => attention mechanism en parallèle
    h = 1   

    d_q, d_k, d_v = 24, 24, d
    
    stacked_inputs = embedded_sentence.repeat(h, 1, 1)
    print("stacked inputs shape :",stacked_inputs.shape)

    multiheaded_attention_layer = MultiHeadedAttentionLayer(h,d,d_q,d_k,d_v)
    multiheaded_attention_layer.processInputs(stacked_inputs, stacked_inputs, stacked_inputs)
    
    multihead_context_vectors = multiheaded_attention_layer.output
    
    print("multihead_context_vectors" , multihead_context_vectors)
    
    #ADD & Norm layer
    #Add vectors and norm layer
    
    pass

        