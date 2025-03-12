'''
Created on 6 mars 2025

@author: SSM9
'''

import torch
import torch.nn as nn
from tutoSelfAttentionMechanism import SelfAttention_v2

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),diagonal=1)
            )
        
    #La méthode forward est implémentée dans la classe mais est appelée par les méthodes internes de nn.Module (probablement la __call__)
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
            )
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec

if __name__ == '__main__':
    
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your (x^1)
         [0.55, 0.87, 0.66], # journey (x^2)
         [0.57, 0.85, 0.64], # starts (x^3)
         [0.22, 0.58, 0.33], # with (x^4)
         [0.77, 0.25, 0.10], # one (x^5)
         [0.05, 0.80, 0.55]] # step (x^6)
        )
    
    d_in = inputs.shape[1]
    d_out = 2
    
    #On réutilise les Wkey et Query de la classe SelfAttention
    
    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print(sa_v2(inputs))
    
    #On calcule les attention weights comme dans un self attention mechanism
    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs)
    attn_scores = queries @ keys.T
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) 
    print(attn_weights)
    
    #On construit le masque (une matrice de 1 dont les valeurs au dela de la diagonale sont nulles)
    #On utilise la méthode Trom pour cela
    context_length = attn_weights.shape[0] 
    mask_simple = torch.tril(torch.ones(context_length, context_length)) 
    print(mask_simple)
    
    #On applique le masque attention weights en multipliant chaque poid par un élément du masque correspondant
    masked_simple = attn_weights*mask_simple
    print(masked_simple)
    
    #On normalise les poids masqué
    row_sums = masked_simple.sum(dim=-1, keepdim=True) 
    masked_simple_norm = masked_simple / row_sums 
    print(masked_simple_norm)
    
    #On peut améliorer le masquage en utilisant des valeurs -inf pour masquer les scores d'attention qu'on souhaite éliminer
    #On calcule une matrice triangulaire supérieur de 1
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1) 
    #On remplace les elements de la matrice attention score par sa valeur si la valeur dans mask = 0 ou -inf si = 1
    masked = attn_scores.masked_fill(mask.bool(), -torch.inf) 
    print(masked)
    
    #Ensuite on normalise cette matrice de score d'attention les score égaux à -inf deviendront des 0
    attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1) 
    print(attn_weights)
    
    #3.5.2 dropout => annuler des résultats dans une matrice Input@W <=> désactiver le neurone correspondant à la colonne associé au résultat nul
    torch.manual_seed(123)
    dropout = torch.nn.Dropout(0.5)
    example = torch.ones(6, 6)
    torch.manual_seed(123)
    print(dropout(example))
    torch.manual_seed(123)
    print(dropout(attn_weights))
    
    #Implementing causal attention class
    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape)
    
    torch.manual_seed(123)
    context_length = batch.shape[1]
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    context_vecs = ca(batch)
    print("context_vecs.shape:", context_vecs.shape)
    
    pass