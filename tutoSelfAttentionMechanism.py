'''
Created on 26 févr. 2025

@author: SSM9
'''

import torch
import torch.nn as nn
import tiktoken
import numpy as np
from ArtificialNeuronNetwork.TransformersComponents.EmbeddingLayer import EmbeddingLayer

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
            )
        context_vec = attn_weights @ values
        return context_vec

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
            super().__init__()
            self.W_query = nn.Parameter(torch.rand(d_in, d_out))
            self.W_key = nn.Parameter(torch.rand(d_in, d_out))
            self.W_value = nn.Parameter(torch.rand(d_in, d_out))
            
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
            )
        context_vec = attn_weights @ values
        return context_vec

if __name__ == '__main__':

    '''
    vocab_size = 50257
    output_dim = 3
    
    #text
    x = "Your journey starts with one step"
    
    tokenizer=tiktoken.get_encoding("gpt2")
    
    token_ids = np.array(tokenizer.encode(x))
    
    print("Token IDs:\n", token_ids)
    print("\nInputs shape:\n", torch.tensor(token_ids).shape)
    
    myEmbddingLayer = EmbeddingLayer(vocab_size, output_dim)
    
    embedded_token_IDs = myEmbddingLayer(token_ids)
    print("Embedded Token IDs:\n", embedded_token_IDs)
    '''
    
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your (x^1)
         [0.55, 0.87, 0.66], # journey (x^2)
         [0.57, 0.85, 0.64], # starts (x^3)
         [0.22, 0.58, 0.33], # with (x^4)
         [0.77, 0.25, 0.10], # one (x^5)
         [0.05, 0.80, 0.55]] # step (x^6)
        )
    
    #Example of computation of the context vector for the vector x^2
    x_2 = inputs[1]
    d_in = inputs.shape[1]
    d_out = 2
    
    torch.manual_seed(123)
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    
    query_2 = x_2 @ W_query
    key_2 = x_2 @ W_key
    value_2 = x_2 @ W_value
    print(query_2)
    
    queries = inputs @ W_query
    print("queries.shape:", queries.shape)
    
    keys = inputs @ W_key
    values = inputs @ W_value
    print("keys.shape:", keys.shape)
    print("values.shape:", values.shape)
    
    keys_2 = keys[1]
    attn_score_22 = query_2.dot(keys_2)
    print(attn_score_22)
    
    attn_scores_x2 = query_2 @ keys.T
    print(attn_scores_x2)
    
    d_k = keys.shape[-1]
    attn_weights_x2 = torch.softmax(attn_scores_x2 / d_k**0.5, dim=-1)
    print(attn_weights_x2)
    
    context_vector_x2 = attn_weights_x2 @ values
    print(context_vector_x2)
    
    #Example of computation of the context vector for the complete input
    queries = inputs @ W_query
    print("queries.shape:", queries.shape)
    
    keys = inputs @ W_key
    values = inputs @ W_value
    print("keys.shape:", keys.shape)
    print("values.shape:", values.shape)
    
    attention_scores = queries@keys.T
    print(attention_scores)
    attention_weights = torch.softmax(attention_scores / d_k**0.5, dim=-1)
    print(attention_weights)
    context_vectors = attention_weights@values
    print(context_vectors)
    
    #Same example using the SelfAttentionV1 class
    
    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1(d_in, d_out)
    print(sa_v1(inputs))
    
    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print(sa_v2(inputs))
    
    #Exercise where parameters from model 2 are assigned to model 1
    sa_v1.W_query = torch.nn.Parameter(sa_v2.W_query.weight.T)
    sa_v1.W_key = torch.nn.Parameter(sa_v2.W_key.weight.T)
    sa_v1.W_value = torch.nn.Parameter(sa_v2.W_value.weight.T)
    
    print(sa_v1(inputs))
    
    pass