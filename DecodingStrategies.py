'''
Created on 11 avr. 2025

@author: SSM9
'''

import torch
import matplotlib.pyplot as plt

def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

def print_sampled_tokens(probas):
    
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item()
              for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

if __name__ == '__main__':
    
    #Temperature scaling method to generate text
    
    #Pour le tuto on imagine un vocabulaire de 9 token
    vocab = {
        "closer": 0,
        "every": 1,
        "effort": 2,
        "forward": 3,
        "inches": 4,
        "moves": 5,
        "pizza": 6,
        "toward": 7,
        "you": 8,
        }
    inverse_vocab = {v: k for k, v in vocab.items()}
    
    #On crée un tenseur de sortie d'un modèle renvoyant des logits (probas brutes)
    next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.05, 1.79])
    
    #On utilise la strategie basée sur le greedy decoding pour identifier le token généré
    probas = torch.softmax(next_token_logits, dim=0)
    next_token_id = torch.argmax(probas).item()
    print(inverse_vocab[next_token_id])
    
    #Pour implémenter une stratégie de génération probabiliste on va utiliser la fonction multinomial (changement de policy)
    torch.manual_seed(123)
    next_token_id = torch.multinomial(probas, num_samples=1).item()
    
    #On peut retomber sur le même tirage qu'avec du greedy decoding mais on peut également en tirer d'autre
    print(inverse_vocab[next_token_id])
    
    print_sampled_tokens(probas)
    
    temperatures = [1, 0.1, 5]
    scaled_probas = [softmax_with_temperature(next_token_logits, T)
                     for T in temperatures]
    
    for probas in scaled_probas:
        print("\n")
        print_sampled_tokens(probas)
    
    x = torch.arange(len(vocab))
    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, T in enumerate(temperatures):
        rects = ax.bar(x + i * bar_width, scaled_probas[i],
                       bar_width, label=f'Temperature = {T}')
    ax.set_ylabel('Probability')
    ax.set_xticks(x)
    ax.set_xticklabels(vocab.keys(), rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    pass