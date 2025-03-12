'''
Created on 21 févr. 2025

@author: SSM9
'''

import torch

from ArtificialNeuronNetwork.NeuronNetwork import NeuronNetwork
from ArtificialNeuronNetwork import Activation_functions
from ArtificialNeuronNetwork import Cost_functions

from ArtificialNeuronNetwork.TransformersComponents.EmbeddingLayer import EmbeddingLayer
from ArtificialNeuronNetwork.TransformersComponents.EmbeddingLayer import PositionEmbeddingLayer
from ArtificialNeuronNetwork.TransformersComponents.EmbeddingLayer import Pos_embedding_type

import tutoSlidingWindowDataLoader
from tutoSlidingWindowDataLoader import create_dataloader_v1

import numpy as np

def one_hot_encoding(token_id, vocab_size):
    
    one_encoded_token_id = np.zeros(vocab_size)
    one_encoded_token_id[token_id]=1
    return one_encoded_token_id

if __name__ == '__main__':
    
    input_ids = torch.tensor([2, 3, 5, 1, 2])
    
    vocab_size = 6
    output_dim = 3
    
    torch.manual_seed(123)
    
    #Génère la couche d'embedding => ça revient à faire une couche de neurones FC qui associe un vecteur issu d'un 1-hot encoding  [0--010---0] à une couche de neurones.
    #Cette couche de neurones à au minimum MxN poids où M = vocab size et N=nombre de neurone => dimension de l'espace sémantique
    #cet algo peut être optimisé en supposant que le tokenID permet juste de récupérer l'indice du vecteur dans la table des poids 
    #vector_result = embedding_layer.weight[tokenID].
    #le hic, c'est que l'apprentissage est moins aisé dans ce cas => on peut envisager une double implémentation selon qu'on soit en apprentissage ou execution (idée en l'air)
    
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(embedding_layer.weight)
    
    print(embedding_layer(torch.tensor([3])))
    
    print(embedding_layer(input_ids))
    
    
    #On va tester le position encoding en partant d'un vrai exemple
    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    
    
    with open("E:\\users\\sami\\trash\\the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    max_length = 4
    dataloader = create_dataloader_v1(
            raw_text, batch_size=8, max_length=max_length,
            stride=max_length, shuffle=False
            )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)
    
    #On va "embedder" nos 8 séquences de token en utilisant la couche d'embedding qu'on a défini
    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)
    
    #On va créer les position encoding vector en attribuant un vecteur unique à chaque position dans une séquence pos1 -> [vecteur de taille output dim]...posMAX_INPUT_SIZE-> [vecteur de taille output dim]
    #On obtiendra un tenseur de taille [MAX_INPUT_SIZE, output dim] 
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(pos_embeddings.shape)
    
    #On additionne à chaque batch de token ID, le vecteur associé à sa position dans le batch
    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)
    
    
    
    #Application de l'embedding en utilisant ma propre librairie de réseau de neurones
    
    print("\n\n\nSame tuto with my own ANN library")
    print("\n\n\nInputs token IDs")
    print(input_ids)
    
    vocab_size = 6
    output_dim = 3
    my_embedding_layer = NeuronNetwork(number_of_inputs= vocab_size, 
                 number_of_outputs= output_dim)
    
    my_embedding_layer_class = EmbeddingLayer(vocab_size, output_dim)
    my_embedding_layer_class.embedding_layer = my_embedding_layer
    
    my_one_encoded_inputs = np.array([one_hot_encoding(token_id, vocab_size) for token_id in input_ids])
    
    
    print("\n\n\nOne encoded Inputs token IDs")
    print(my_one_encoded_inputs.shape)
    print(my_one_encoded_inputs)
    
    my_embedded_token_IDs = my_embedding_layer.executeModelOnBatch(my_one_encoded_inputs)
    
    my_embedded_token_IDs_2 = my_embedding_layer_class(input_ids)
    
    print("\n\n\nOne embedded input IDs")
    print(my_embedded_token_IDs.shape)
    print(my_embedded_token_IDs)
    print(my_embedded_token_IDs_2.shape)
    print(my_embedded_token_IDs_2)
    
    my_pos_layer = NeuronNetwork(number_of_inputs= 1, 
                 number_of_outputs= output_dim)
    
    my_pos_layer_class = PositionEmbeddingLayer(5,output_dim,Pos_embedding_type.ABSOLUTE)
    my_pos_layer_class.pos_embedding_layer = my_pos_layer
    
    
    
    print("\n\n\nAbsolute Position vector IDs")
    print(np.resize(np.arange(0,5,1),(5,1)))
    my_pos_embdeddings =  my_pos_layer.executeModelOnBatch(np.resize(np.arange(0,5,1),(5,1)))
    
    print("\n\n\nAbsolute Position vector Embeddings")
    print(my_pos_embdeddings)
    
    
    my_input_embeddings = np.add(my_embedded_token_IDs,my_pos_embdeddings)
    
    my_input_embeddings_2 = my_pos_layer_class(my_embedded_token_IDs_2)
    
    print("\n\n\nInput Embeddings")
    print(my_input_embeddings)
    print(my_input_embeddings_2)
    
    
    
    pass