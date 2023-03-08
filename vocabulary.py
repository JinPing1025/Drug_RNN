
from os import path
import numpy as np


class Vocabulary():
    def __init__(self,max_len = 100):
        self.max_len = max_len
    
    def vocab(self,smiles):
        characters = set()
        for i, mol in enumerate(smiles):
            mol = mol.replace('Br', 'R').replace('Cl', 'L')
            for char in mol:
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
        smile_data = []
        for mol in data:
            mol = mol.replace('Br', 'R').replace('Cl', 'L')
            mol_token = []
            mol_token.append('G')
            for char in mol:
                mol_token.append(char)
            if len(mol_token) < self.max_len:
                a = self.max_len - len(mol_token)
                [mol_token.append('A') for _ in range(a)]
            smile_data.append(mol_token)
        return smile_data
    
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

