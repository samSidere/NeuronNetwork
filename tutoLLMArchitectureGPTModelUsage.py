'''
Created on 26 mars 2025

@author: SSM9
'''

import torch
import torch.nn as nn

import tiktoken

from ArtificialNeuronNetwork.GPTModel import GPTModel

if __name__ == '__main__':
    
    #Dictionnaire utilisé pour configurer les paramètres du modèle
    GPT_CONFIG_124M = {
        "vocab_size": 50257, # Vocabulary size
        "context_length": 1024, # Context length
        "emb_dim": 768, # Embedding dimension
        "n_heads": 12, # Number of attention heads
        "n_layers": 12, # Number of layers
        "drop_rate": 0.1, # Dropout rate
        "qkv_bias": False # Query-Key-Value bias
        }
    
    GPT_CONFIG_medium = {
        "vocab_size": 50257, # Vocabulary size
        "context_length": 1024, # Context length
        "emb_dim": 1024, # Embedding dimension
        "n_heads": 16, # Number of attention heads
        "n_layers": 24, # Number of layers
        "drop_rate": 0.1, # Dropout rate
        "qkv_bias": False # Query-Key-Value bias
        }
    
    GPT_CONFIG_large = {
        "vocab_size": 50257, # Vocabulary size
        "context_length": 1024, # Context length
        "emb_dim": 1280, # Embedding dimension
        "n_heads": 20, # Number of attention heads
        "n_layers": 36, # Number of layers
        "drop_rate": 0.1, # Dropout rate
        "qkv_bias": False # Query-Key-Value bias
        }
    
    GPT_CONFIG_xl = {
        "vocab_size": 50257, # Vocabulary size
        "context_length": 1024, # Context length
        "emb_dim": 1600, # Embedding dimension
        "n_heads": 25, # Number of attention heads
        "n_layers": 48, # Number of layers
        "drop_rate": 0.1, # Dropout rate
        "qkv_bias": False # Query-Key-Value bias
        }
    
    #Create and encode a batch of inputs (txt1 and txt2)
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch)
    
    #Instanciate a model using GPTModel class and GPT CONFIG
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    
    #Process input batch
    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    print(out)
    
    #Extracts the last vector, which corresponds to the next token that the GPT model is supposed to generate
    out = out[::,out.shape[1]-1:out.shape[1]:,::]
    print("\nOutput shape after next token extraction:", out.shape)
    print(out)
    
    #Converts logits into probability distribution using the softmax function
    out = torch.softmax(out,dim=-1)
    
    #Indentifies the index position of the largest value, which also represents the token ID
    out = torch.argmax(out,dim=-1)
    print("\nOutput shape:", out.shape)
    print(out)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    
    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)
    
    #Weight Tying improves the performance of language models by tying (sharing) the weights of the embedding and softmax layers. 
    total_params_gpt2 = (
        total_params - sum(p.numel()
                           for p in model.out_head.parameters())
        )
    print(f"Number of trainable parameters "
          f"considering weight tying: {total_params_gpt2:,}"
          )
    
    mha_params = sum(p.numel() for trf_block in model.trf_blocks for p in trf_block.att.parameters() )/GPT_CONFIG_124M["n_layers"]
    ff_params = sum(p.numel() for trf_block in model.trf_blocks for p in trf_block.ff.parameters() )/GPT_CONFIG_124M["n_layers"]
    
    
    print(f"Total number of parameters in mha : {mha_params:,} and ff : {ff_params:,}")
    
    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} MB")
    
    #Exercice 4.1
    '''
    model_medium = GPTModel(GPT_CONFIG_medium)
    model_large = GPTModel(GPT_CONFIG_large)
    model_xl = GPTModel(GPT_CONFIG_xl)
    
    total_params_gpt2_medium = sum(p.numel() for p in model_medium.parameters())
    total_params_gpt2_large = sum(p.numel() for p in model_large.parameters())
    total_params_gpt2_xl = sum(p.numel() for p in model_xl.parameters())
    print(f"Total number of parameters (medium): {total_params_gpt2_medium:,}")
    print(f"Total number of parameters (large): {total_params_gpt2_large:,}")
    print(f"Total number of parameters (xl): {total_params_gpt2_xl:,}")
    '''
    
    pass