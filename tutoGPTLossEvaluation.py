'''
Created on 3 avr. 2025

@author: SSM9
'''


import torch
import torch.nn as nn

import torch.nn.functional as F

import tiktoken

from ArtificialNeuronNetwork.GPTModel import GPTModel
from ArtificialNeuronNetwork.GPTModel import create_dataloader_v1
from ArtificialNeuronNetwork.GPTModel import calc_loss_loader

#The function gets an token sequence as an input and generate a sequence of token appended to the input
def generate_text_simple(model, idx, max_new_tokens, context_size):
    
    for _ in range(max_new_tokens):
        #Get the last context_size tokens from the input to feed the model for each sentence of the batch
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            #Feed model 
            logits = model(idx_cond)
            
            #Extracts the last vector, which corresponds to the next token that the GPT model is supposed to generate
            logits = logits[:, -1, :]
            
            #Converts logits into probability distribution using the softmax function
            probas = torch.softmax(logits, dim=-1)
            
            #Identifies the index position of the largest value, which also represents the token ID
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)
            
            #Concatenate output token with the input sentence to generate text
            idx = torch.cat((idx, idx_next), dim=1)
            
    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    
    #.unsqueeze(0) adds the batch dimension
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    
    #Squeeze 0 Removes batch dimension
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

if __name__ == '__main__':
    
    #GPT Model config in a dict
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    #Init GPT Model
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    
    #Create context sentence used as an input
    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")
    
    #Convert text to token ids sequence
    input_tokenid_sequence =text_to_token_ids(start_context, tokenizer)
    
    generated_token_ids = generate_text_simple(
        model=model,
        idx=input_tokenid_sequence,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"])
    
    generated_text = token_ids_to_text(generated_token_ids, tokenizer)
    
    print("Output text:\n", generated_text)
    
    #Let us evaluate model performance    
    
    #Fake dataset declaration
    inputs = torch.tensor([[16833, 3626, 6100], # ["every effort moves",
                           [40, 1107, 588]]) # "I really like"]
    
    targets = torch.tensor([[3626, 6100, 345 ], # [" effort moves you",
                            [1107, 588, 11311]]) # " really like chocolate"]
    
    
    with torch.no_grad():
        logits = model(inputs)
        
    output_probas = torch.softmax(logits, dim=-1)
    print(output_probas.shape)
    
    token_ids = torch.argmax(output_probas, dim=-1, keepdim=True)
    print("output Token IDs:\n",token_ids)
    
    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    print(f"Outputs batch 1:"
          f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
        
    
    text_idx = 0
    #Genere une table qui donne la probabilité d'avoir le token ciblé pour chaque token généré de la phrase 1 (entrée).
    target_probas_1 = output_probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("probas of having the right token id for each token of Text 1:\n", target_probas_1,"\n")
    text_idx = 1
    #Genere une table qui donne la probabilité d'avoir le token ciblé pour chaque token généré de la phrase 2 (entrée).
    target_probas_2 = output_probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("probas of having the right token id for each token of Text 2\n:", target_probas_2,"\n")
    
    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print(log_probas)
    
    avg_log_probas = torch.mean(log_probas)
    print(avg_log_probas)
    
    neg_avg_log_probas = avg_log_probas * -1
    print(neg_avg_log_probas)
    
    print("Logits shape:", logits.shape)
    print("Targets shape:", targets.shape)
    
    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    print("Flattened logits:", logits_flat.shape)
    print("Flattened targets:", targets_flat.shape)
    
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print(loss)
    
    #Convert targets into proba vectors
    targets_probas = F.one_hot(targets, GPT_CONFIG_124M["vocab_size"]).type(torch.float)
    
    #Computation algo before tuto
    #flatten reduces the number of dimension of the tensor by flattening it. In this case, we want to flatten the tensor by concatenating the dim 0 tensors
    output_probas = output_probas.flatten(0, 1)
    targets_probas = targets_probas.flatten(0, 1)
    print("output_probas shape:", output_probas.shape)
    print("targets_probas shape:", targets_probas.shape)
    
    #In this case we didn't take into account the torch cross entropy method power (it can handle batch flattened output logits and target token ids directly)
    loss = F.cross_entropy(input=output_probas, target=targets_probas)
    
    print("loss function result is :",loss)
    
    #Calculating the training and validation set losses : Utilisation d'un dataset pour entrainer le model
    file_path = "E:\\users\\sami\\trash\\the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
        
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print("Characters:", total_characters)
    print("Tokens:", total_tokens)
    
    #On va définir un train ration qui va nous donner la taille du set d'entrainement versus le set de test
    train_ratio = 0.90
    
    #On calcule l'index qui définira la fontière entre dataset d'entrainement et de test
    split_idx = int(train_ratio * len(text_data))
    
    #on slice le dataset pour séparer les données d'entrainement et de test
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    
    torch.manual_seed(123)
    #On crée les dataloader de test et d'entrainement
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
        )
    
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
        )
    
    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)
        
    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)
        
    #On va calculer les performances du model sur un vrai dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #If you have a machine with a CUDA-supported GPU, the LLM will train on the GPU without making any changes to the code.
    model.to(device)
    
    #Disables gradient tracking for efficiency because we are not training yet
    with torch.no_grad():
        #Via the “device” setting, we ensure the data is loaded onto the same device as the LLM model. 
        #train_loss = calc_loss_loader(train_loader, model, device)
        #val_loss = calc_loss_loader(val_loader, model, device)
        train_loss = model.calc_loss_loader(train_loader, device)
        val_loss = model.calc_loss_loader(val_loader, device)
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)
    
    pass