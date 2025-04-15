'''
Created on 9 avr. 2025

@author: SSM9
'''

import torch
import torch.nn as nn

import torch.nn.functional as F

import tiktoken

from ArtificialNeuronNetwork.GPTModel import GPTModel
from ArtificialNeuronNetwork.GPTModel import create_dataloader_v1
from ArtificialNeuronNetwork.GPTModel import train_model_simple

from ArtificialNeuronNetwork.GPTModel import generate_text_simple
from ArtificialNeuronNetwork.GPTModel import text_to_token_ids
from ArtificialNeuronNetwork.GPTModel import token_ids_to_text

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label="Validation loss"
        )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    #Creates a second x-axis that shares the same y-axis
    ax2 = ax1.twiny()
    
    #Invisible plot for aligning ticks
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

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
    
    tokenizer = tiktoken.get_encoding("gpt2")
    #On va calculer les performances du model sur un vrai dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    #Calculating the training and validation set losses : Utilisation d'un dataset pour entrainer le model
    '''
    file_path = "E:\\users\\sami\\trash\\the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    '''
    file_path1 = "E:\\users\\sami\\trash\\fr_text1.txt"
    file_path2 = "E:\\users\\sami\\trash\\fr_text2.txt"
    file_path3 = "E:\\users\\sami\\trash\\fr_text3.txt"
    with open(file_path1, "r", encoding="utf-8") as file:
        text_data1 = file.read()
    with open(file_path2, "r", encoding="utf-8") as file:
        text_data2 = file.read()
    with open(file_path3, "r", encoding="utf-8") as file:
        text_data3 = file.read()
        
    text_data = " <|end_of_text|> ".join((text_data1, text_data2, text_data3))
    #'''    
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    
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
    
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.0004, weight_decay=0.1)
    num_epochs = 10
    
    #'''
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Chaque fois qu'elle", tokenizer=tokenizer
    )
    '''
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )
    '''
    
    #epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    #plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    
    model.to("cpu")
    model.eval()
    
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids("Chaque fois qu'elle", tokenizer),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"]
        )
    '''
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"]
        )
    '''
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
    
    #Temperature scaling method to generate text
    
    pass