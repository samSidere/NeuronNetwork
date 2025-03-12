'''
Created on 26 févr. 2025

@author: SSM9
'''

import torch
import tiktoken
import numpy as np
from ArtificialNeuronNetwork.TransformersComponents.EmbeddingLayer import EmbeddingLayer

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

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
    query = inputs[1]
    attn_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, query)
    print(attn_scores_2)
    
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
    print("Attention weights:", attn_weights_2_tmp)
    print("Sum:", attn_weights_2_tmp.sum())
    
    attn_weights_2_naive = softmax_naive(attn_scores_2)
    print("Attention weights:", attn_weights_2_naive)
    print("Sum:", attn_weights_2_naive.sum())
    
    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
    print("Attention weights:", attn_weights_2)
    print("Sum:", attn_weights_2.sum())
    
    context_vector_2 = torch.empty(query.shape[0])
    inputsT = torch.t(inputs)

    for i, x_iT in enumerate(inputsT):
        context_vector_2[i] = torch.dot(x_iT, attn_weights_2)
    print(context_vector_2)
    
    #Example of computation of the context vector for the complete entry
    attn_scores = torch.empty(inputs.shape[0],inputs.shape[0])
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores[i,j] = torch.dot(x_i, x_j)
    print(attn_scores)
    
    attn_scores = inputs @ inputs.T
    print(attn_scores)
    
    attn_weights = torch.softmax(attn_scores, dim=-1)
    print(attn_weights)
    
    row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
    print("Row 2 sum:", row_2_sum)
    print("All row sums:", attn_weights.sum(dim=-1))
    
    all_context_vecs = attn_weights @ inputs
    print(all_context_vecs)
    
    print("Previous 2nd context vector:", context_vector_2)
    
    pass