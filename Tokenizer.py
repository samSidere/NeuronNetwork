'''
Created on 11 févr. 2025

@author: SSM9
'''

import re

class SimpleTokenizerV1(object):
    '''
    classdocs
    '''
    
    vocab = None
    from_id_to_text = None
    from_text_to_id = None
    

    def __init__(self, vocab):
        self.vocab = vocab
        self.from_text_to_id = vocab
        self.from_id_to_text = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        
        #Transformation d'une phrase en série de token
        tokenized_text = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokenized_text = [item.strip() for item in tokenized_text if item.strip()]
        
        
        token_ids_list = [ self.from_text_to_id[token] for token in tokenized_text]
        return token_ids_list

    def decode (self, token_ids_list): 
        
        text = " ".join([self.from_id_to_text[i] for i in token_ids_list])
        text = re.sub(r'\s+([,.:;?_!"()\']|--|\s)', r'\1', text)
        #text = re.sub(r'(--)+\s', r'\1', text)
        
        return text

#On va surcharger la classe SimpleTokenizer V1 pour gérer les tokens qui ne font pas partie du vocabulaire
class SimpleTokenizerV2(SimpleTokenizerV1):
    '''
    classdocs
    '''

    def __init__(self, vocab):
        
        SimpleTokenizerV1.__init__(self, vocab)
    
    #On surcharge encode pour remplacer les tokens inconnus par <|unk|>
    def encode(self, text):
        
        #Transformation d'une phrase en série de token
        tokenized_text = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokenized_text = [item.strip() for item in tokenized_text if item.strip()]
        
        #On remplace les tokens inconnus par <|unk|>
        tokenized_text = [item if item in self.from_text_to_id else "<|unk|>" for item in tokenized_text]
        
        token_ids_list = [ self.from_text_to_id[token] for token in tokenized_text]
        return token_ids_list

