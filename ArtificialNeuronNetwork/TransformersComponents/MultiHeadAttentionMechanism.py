'''
Created on 25 mars 2025

@author: SSM9
'''

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_in, d_out,
                 context_length, dropout, num_heads, qkv_bias=False):
        
        super().__init__()
        assert (d_out % num_heads == 0), \
        "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
            )
        
    def forward(self, x):
        
        b, num_tokens, d_in = x.shape
        
        #On fait passer l'entrée dans les 3 couches linéaires
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        #On réarrange les dimensions des tenseurs pour séparer les têtes (probablement pour les normalisations qui vont suivre)
        #We implicitly split the matrix by adding a num_heads dimension. 
        #Then we unroll the last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim).
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        
        #la transposition permet de finir de séparer les têtes
        #Transposes from shape (b, num_tokens,num_heads, head_dim) to (b, num_heads,num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        
        #Computes dot product for each head
        attn_scores = queries @ keys.transpose(2, 3)
        
        #Masks truncated to the number of tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        
        #Uses the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        #Normalisation des scores d'attention de toutes les têtes pour obtenir les poids d'attention (attention weights)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        
        #Application du dropout à chaque matrice de scores d'attention
        attn_weights = self.dropout(attn_weights)
        
        #Tensor shape: On fait le produit matricielle qui aura une shape (b, n_heads, num_tokens, head_dim) 
        #puis on transpose ->  (b, num_tokens, n_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        
        #On retourne au format concatené (b, num_tokens, n_heads, head_dim) -> (b, num_tokens, dout)
        #Combines heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        
        #Adds an optional linear projection => On passe dans une couche linéaire dout-> dout
        context_vec = self.out_proj(context_vec)
        
        return context_vec