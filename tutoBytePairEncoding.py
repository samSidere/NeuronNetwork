'''
Created on 18 févr. 2025

@author: SSM9
'''

from importlib.metadata import version
import tiktoken

if __name__ == '__main__':
    
    
    print("tiktoken version:", version("tiktoken"))
    tokenizer = tiktoken.get_encoding("gpt2")
    
    text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces"
            "of someunknownPlace.")
    
    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)
    
    strings = tokenizer.decode(integers)
    print(strings)
    
    text = ("Akwirwier")
    
    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)
    
    strings = tokenizer.decode(integers)
    print(strings)
    
    strings = [tokenizer.decode([tokenID]) for tokenID in integers]
    print(strings)
    
    
    pass