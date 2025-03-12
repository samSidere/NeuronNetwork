'''
Created on 3 sept. 2024

@author: SSM9
'''

import numpy as np
import torch
import re
from Tokenizer import SimpleTokenizerV1
from Tokenizer import SimpleTokenizerV2

def iterativeParsing(array):
    print("work on"+str(array))
    if array.ndim > 1 :
        for arr in array:
            iterativeParsing(arr)
    else :
        print("Pixel content is"+str(array))  

if __name__ == '__main__':
    
    '''
        Reflection sur les réseaux de neurones 
    '''
    '''
        Code pour tester les techniques de tokenisation de texte
    '''
    
    #On récupère un texte d'internet
    with open("E:\\users\\sami\\trash\\the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read() 
    
    print("Total number of character:", len(raw_text))
    print(raw_text[:99])
    
    #Exemple de tokenization simple
    text = "Hello, world. Is this-- a test?"
    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    result = [item.strip() for item in result if item.strip()]
    print(result)
    
    #tokenization du livre
    #On récupère les données, on casse ensuite ce texte pour en faire une structure de données (tableau)
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print(len(preprocessed))
    print(preprocessed[:30])
    
    #Creation du dictionnaire => mapping tokens <-> token ID
    #On conserve les première itération des mots dans le texte. Ensuite, on trie les mots contenue dans le texte
    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    print("vocab size based on book content ",vocab_size)
    #print(all_words[:30])
    
    #Extension of the vocabulary to manage End Of Text and Unknown tokens
    all_words.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token:integer for integer,token in enumerate(all_words)}
    print("vocab size based on book content + vocab extension ",len(vocab.items()))
    
    #enumerate => transforme un tableau en énumerré
    #vocabulaire => simply cast an array into an enumerate and store the result into an object     
    vocab = {token:integer for integer,token in enumerate(all_words)}
    
    for i, item in enumerate(list(vocab.items())[-5:]):
        print(item)
    
    #print(vocab.items())
    for i, item in enumerate(vocab.items()):
        #print(item)
        if i in [53,150,115,256,486]:
            print(item)
            #break
    
    #On va tokeniser une nouvelle phrase
    sentence = "I had am a cheap genius HAD"
    
    #Transformation d'une phrase en série de token
    tokenized_sentence = re.split(r'([,.:;?_!"()\']|--|\s)', sentence)
    
    tokenized_sentence = [item.strip() for item in tokenized_sentence if item.strip()]
    
    #print(sentence)
    print(tokenized_sentence)
    
    #On va générer la liste des token ID associé à chaque token de la phase
    tokenID_sentence = [ vocab[token] for token in tokenized_sentence]
    print(tokenID_sentence)
    
    #On va tokeniser le livre
    tokenID_preprocessed = [ vocab[token] for token in preprocessed]
    print(tokenID_preprocessed[:30])
    
    #Utilisation du SimpleTokenizeV1 pour transformer le livre en liste de tokenID => Embedding
    myTokenizer = SimpleTokenizerV1(vocab)
    tokenID_sentence = myTokenizer.encode(raw_text)
    print(tokenID_sentence[:30])
    
    #Utilisation du SimpleTokenizeV1 pour décoder le livre
    print(myTokenizer.decode(tokenID_sentence))
        
    #Autres exemples
    text = """"It's the last he painted, you know,"
        Mrs. Gisburn said with pardonable pride."""
    ids = myTokenizer.encode(text)
    print(ids)
    print(myTokenizer.decode(ids))
    
    text1 = "Hello, do you like tea?"
    #print(myTokenizer.encode(text))
    
    #On va apprendre à gérer les mots inconnus dans le vocabulaire
    myTokenizer2 = SimpleTokenizerV2(vocab)
    
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(text)
    tokenID_sentence = myTokenizer2.encode(text)
    print(tokenID_sentence)
    print(myTokenizer2.decode(tokenID_sentence))
    
    
    
    pass