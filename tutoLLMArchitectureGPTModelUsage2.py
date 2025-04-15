'''
Created on 26 mars 2025

@author: SSM9
'''

import torch
import torch.nn as nn

import tiktoken

from ArtificialNeuronNetwork.GPTModel import GPTModel

#A get next token function
def getNextToken(input, model):
    
    logits = model(input)
    
    #Extracts the last vector, which corresponds to the next token that the GPT model is supposed to generate
    logits = logits[::,-1::,::]
    
    #Converts logits into probability distribution using the softmax function
    probas = torch.softmax(logits,dim=-1)
    
    #Identifies the index position of the largest value, which also represents the token ID
    next_token_ids = torch.argmax(probas,dim=-1)
    
    return next_token_ids

#Exemple from the lesson
def generate_text_simple(model, idx, max_new_tokens, context_size):
    
    for _ in range(max_new_tokens):
        #Get the last context_size tokens from the input to feed the model for each sentence of the batch
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            #Feed model 
            logits = model(idx_cond)
            logits = logits[:, -1, :]
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx

if __name__ == '__main__':
    
    #Dictionnaire utilisé pour configurer les paramètres du modèle
    GPT_CONFIG_124M = {
        "vocab_size": 50257, # Vocabulary size
        "context_length": 1024, # Context length
        "emb_dim": 768, # Embedding dimension
        "n_heads": 12, # Number of attention heads
        "n_layers": 12, # Number of layers
        "drop_rate": 0.1, # Dropout rate
        "drop_rate_embedding": 0.1, # Dropout rate
        "drop_rate_transformer": 0.1, # Dropout rate
        "drop_rate_attention": 0.1, # Dropout rate
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
    
    #Let us use the previous code wrapped in a function in order to generate text from context 10 times
    batch = []
    txt = "Hello, I am"
    batch.append(torch.tensor(tokenizer.encode(txt)))
    batch = torch.stack(batch, dim=0)
    print(batch)
    
    model.eval()
    for _ in range (0,6,1):
        
        batch = torch.cat((batch,getNextToken(batch, model)),dim=-1)
        print (batch)
    
    
    #Let s generate text using generate_text_simple function
    
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)
    
    
    model.eval()
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
        )
    print("Output:", out)
    print("Output length:", len(out[0]))
    
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)
    
    pass