'''
Created on 11 mars 2025

@author: SSM9
'''

import torch
import torch.nn as nn
from tutoCausalAttentionMechanism import CausalAttention

#Let's move this class to a proper library
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

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(
                d_in, d_out, context_length, dropout, qkv_bias
                )
                for _ in range(num_heads)]
            )
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

if __name__ == '__main__':
    
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your (x^1)
         [0.55, 0.87, 0.66], # journey (x^2)
         [0.57, 0.85, 0.64], # starts (x^3)
         [0.22, 0.58, 0.33], # with (x^4)
         [0.77, 0.25, 0.10], # one (x^5)
         [0.05, 0.80, 0.55]] # step (x^6)
        )
    
    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape)
    
    torch.manual_seed(123)
    context_length = batch.shape[1] # This is the number of tokens
    d_in, d_out = 3, 2
    
    #Example of multihead where the class is wrapping several singlehead attention mechanisms working in parallel
    mha = MultiHeadAttentionWrapper(
        d_in, d_out, context_length, 0.0, num_heads=2
        )
    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)
    
    #Example of multihead optimized using matrix mul
    torch.manual_seed(123)
    batch_size, context_length, d_in = batch.shape
    d_out = 4
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)
    
    #Exercize example
    mha2 = MultiHeadAttention(768, 768, 1024, 0.0, num_heads=12)
    
    pass