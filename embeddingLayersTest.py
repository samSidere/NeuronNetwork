'''
Created on 25 févr. 2025

@author: SSM9
'''
import torch

from ArtificialNeuronNetwork.NeuronNetwork import NeuronNetwork
from ArtificialNeuronNetwork import Activation_functions
from ArtificialNeuronNetwork import Cost_functions

from ArtificialNeuronNetwork.TransformersComponents.EmbeddingLayer import EmbeddingLayer
from ArtificialNeuronNetwork.TransformersComponents.EmbeddingLayer import PositionEmbeddingLayer
from ArtificialNeuronNetwork.TransformersComponents.EmbeddingLayer import Pos_embedding_type

from ArtificialNeuronNetwork.TransformersComponents.TokenDataLoader import GPTDataloader_v1

if __name__ == '__main__':
    
    
    vocab_size = 50257
    output_dim = 50
    
    #Open document used as data source
    with open("E:\\users\\sami\\trash\\the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    #Create Data loader allowing to create batches of data from text
    #Also manage text->tokenIds conversion
    max_length = 4
    my_data_loader = GPTDataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
    
    inputs, targets = my_data_loader()
    
    inputs, targets = my_data_loader()
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)
    
    #token embedding
    
        #torch method
    token_embedding_layer_torch = torch.nn.Embedding(vocab_size, output_dim)
    token_embeddings_torch = token_embedding_layer_torch(inputs)
    print("\nToken embedding shape torch:\n", token_embeddings_torch.shape)
    #print("\nToken embedding torch:\n", token_embeddings_torch)
    
        #my method
    token_embedding_layer = EmbeddingLayer(vocab_size, output_dim)
    token_embeddings = token_embedding_layer(inputs)
    print("\nToken embedding shape:\n", token_embeddings.shape)
    #print("\nToken embedding:\n", token_embeddings)
    
    #position embedding
    
    context_length = max_length
    
        #torch method
    pos_embedding_layer_torch = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings_torch = pos_embedding_layer_torch(torch.arange(context_length))
    print("\nPosition embedding shape torch:\n", pos_embeddings_torch.shape)
    #print("\nPosition embedding torch:\n", pos_embeddings_torch)
    
    input_embeddings_torch = token_embeddings_torch + pos_embeddings_torch
    print("\nInputs embedding shape torch:\n", input_embeddings_torch.shape)
    #print("\nInputs embedding torch:\n", input_embeddings_torch)
    
        #my method
    pos_embedding_layer = PositionEmbeddingLayer(context_length,output_dim,Pos_embedding_type.ABSOLUTE)
    input_embeddings = pos_embedding_layer(token_embeddings)
    print("\nInputs embedding shape:\n", input_embeddings.shape)
    #print("\nInputs embedding:\n", input_embeddings)
    
    pass