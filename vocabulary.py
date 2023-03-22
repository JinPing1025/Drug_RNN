
from os import path
import numpy as np
import re

class Vocabulary():
    def __init__(self,max_len = 100):
        self.max_len = max_len
    
    def vocab(self,smiles):
        regex = '(\[[^\[\]]{1,10}\])'
        characters = set()
        for i, sm in enumerate(smiles):
            sm = sm.replace('Br', 'R').replace('Cl', 'L')
            sm_chars = re.split(regex, sm)
            for section in sm_chars:
                if section.startswith('['): 
                    characters.add(section)
                else:
                    for char in section:
                        characters.add(char)
        
        characters.add('G')
        characters.add('A')
        unique_character = sorted(characters)

        print('Number of unique characters in the vocabulary: {} '.format(len(unique_character)))
        print(unique_character)
        vocabs = sorted(list(unique_character))
        
        self.char_to_int = dict()
        for i, char in enumerate(vocabs):
            self.char_to_int[char] = i
            
        self.int_to_char = dict()
        for i, char in enumerate(vocabs):
            self.int_to_char[i] = char  
        
        vocab_size = len(vocabs)
        return vocabs,self.char_to_int,self.int_to_char
    
    def tokenize(self,data):
        list_tok_smiles = []  
        for smile in data:
            regex = '(\[[^\[\]]{1,10}\])'
            smile = smile.replace('Br', 'R').replace('Cl', 'L')
            smile_chars = re.split(regex, smile)
            smile_tok = []
            smile_tok.append('G')      
            for section in smile_chars:
                if section.startswith('['): 
                    smile_tok.append(section)
                else:      
                    for char in section:
                        smile_tok.append(char)
            
            smile_tok.append('A')
            if len(smile_tok) > self.max_len:
                continue
            if len(smile_tok) < self.max_len:
                dif = self.max_len - len(smile_tok)
                [smile_tok.append('A') for _ in range(dif)]
            assert len(smile_tok) == self.max_len 
            list_tok_smiles.append(smile_tok)
        return(list_tok_smiles)

    def encode(self,smile_data):
        encode_smiles = []
        for smile in smile_data:
            smile_idx = []
            for char in smile:
                smile_idx.append(self.char_to_int[char])
            encode_smiles.append(smile_idx)
        return encode_smiles
    
    def decode(self,encode_smiles):
        decode_smiles = []
        for smile in encode_smiles:
            smile_chars = []
            for i in smile:
                if (self.int_to_char[i] == 'G'):
                    continue
                if (self.int_to_char[i] == 'A'):
                    break
                smile_chars.append(self.int_to_char[i])
            smile_str = ''.join(smile_chars)
            smile_str = smile_str.replace('R', 'Br').replace('L', 'Cl')
            decode_smiles.append(smile_str)
        return decode_smiles
    
    def creat_label(self,data):
        dataY = [line[1:] for line in data] 
        for i in range(len(dataY)):
            dataY[i].append(self.char_to_int['A'])
        return dataY












