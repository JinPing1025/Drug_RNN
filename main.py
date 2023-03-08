
from os import path
import numpy as np
import re

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"


# def vocab(smiles):
#     characters = set()
#     for i, mol in enumerate(smiles):
#         mol = mol.replace('Br', 'R').replace('Cl', 'L')
#         for char in mol:
#             characters.add(char)
#     characters.add('G')
#     characters.add('A')
#     unique_character = sorted(characters)
#     print('Number of unique characters in the vocabulary: {} '.format(len(unique_character)))
#     print(unique_character)
#     vocabs = sorted(list(unique_character))
    
#     char_to_int = dict()
#     for i, char in enumerate(vocabs):
#         char_to_int[char] = i
        
#     int_to_char = dict()
#     for i, char in enumerate(vocabs):
#         int_to_char[i] = char  
    
#     vocab_size = len(vocabs)
#     return vocabs,char_to_int,int_to_char


def vocab(smiles):
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
    
    char_to_int = dict()
    for i, char in enumerate(vocabs):
        char_to_int[char] = i
        
    int_to_char = dict()
    for i, char in enumerate(vocabs):
        int_to_char[i] = char  
    
    vocab_size = len(vocabs)
    return vocabs,char_to_int,int_to_char
    

# def tokenize(data):
#     smile_data = []
#     for mol in data:
#         mol = mol.replace('Br', 'R').replace('Cl', 'L')
#         mol_token = []
#         mol_token.append('G')
#         for char in mol:
#             mol_token.append(char)
#         if len(mol_token) < max_len:
#             a = max_len - len(mol_token)
#             [mol_token.append('A') for _ in range(a)]
#         smile_data.append(mol_token)
#     return smile_data


def tokenize(data):
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
        if len(smile_tok)>max_len:
            continue
        if len(smile_tok) <max_len:
              dif = max_len - len(smile_tok)
              [smile_tok.append('A') for _ in range(dif)]
                    
        assert len(smile_tok) == max_len 
        list_tok_smiles.append(smile_tok)
    return(list_tok_smiles)


def encode(smile_data):
    encode_smiles = []
    for smile in smile_data:
        smile_idx = []
        for char in smile:
            smile_idx.append(char_to_int[char])
        encode_smiles.append(smile_idx)
    return encode_smiles


def decode(encode_smiles):
    decode_smiles = []
    for smile in encode_smiles:
        smile_chars = []
        for i in smile:
            if (int_to_char[i] == 'G'):
                continue
            if (int_to_char[i] == 'A'):
                break
            smile_chars.append(int_to_char[i])
        smile_str = ''.join(smile_chars)
        smile_str = smile_str.replace('R', 'Br').replace('L', 'Cl')
        decode_smiles.append(smile_str)
    return decode_smiles


def creat_label(data):
    dataY = [line[1:] for line in data] 
    for i in range(len(dataY)):
        dataY[i].append(char_to_int['A'])
    return dataY


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense, Input,GRU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow import keras


import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.layers import concatenate


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

def gelu(x):
    return 0.5*x*(1+tf.keras.activations.tanh(np.sqrt(2/np.pi)*(x+0.044715*pow(x,3))))

# def generator_model():
#     z = Input(x_shape)
#     x = Embedding(vocab_size, emb_dim, input_length = max_len)(z)
#     x = GRU(512, return_sequences=True)(x)
#     x = Dropout(0.5)(x)
    
#     x = GRU(512, return_sequences=True)(x)
#     x = Dropout(0.5)(x)
    
#     x = GRU(512, return_sequences=True)(x)
#     x = Dropout(0.5)(x)
    
#     x = Dense(vocab_size, activation = 'softmax')(x)
    
#     return keras.Model(inputs=z, outputs=x, name="Generator")

from nested_lstm import NestedLSTM

def generator_model():
    z = Input(x_shape)
    x1 = Embedding(vocab_size, emb_dim, input_length = max_len)(z)
    x2 = Embedding(vocab_size, emb_dim, input_length = max_len)(z)
    x1 = NestedLSTM(512, depth=2, dropout=0.5, recurrent_dropout=0.2)(x1)
    x2 = NestedLSTM(512, depth=2, dropout=0.5, recurrent_dropout=0.2)(x2)

    x = concatenate([x1,x2])
    x = Dropout(0.5)(x)
    x = Dense(vocab_size, activation = 'softmax')(x)
    
    return keras.Model(inputs=z, outputs=x, name="Generator")

# def generator_model():
#     z = Input(x_shape)
#     x = Embedding(vocab_size, emb_dim, input_length = max_len)(z)
#     x = NestedLSTM(512, depth=2, dropout=0.5, recurrent_dropout=0.2)(x)
    
#     x = Dense(vocab_size, activation = 'softmax')(x)
    
#     return keras.Model(inputs=z, outputs=x, name="Generator")

def generate_sample(start_idx, num, end_idx,Models):
    seq_list = []
    for j in tqdm(range(num)):
        seq = [start_idx]
        for i in range(max_len-1):
            x = np.reshape(seq, (1, len(seq),1))
            preds = Models.predict(x, verbose=1)    #(1,1,33)
            index = sample(preds[0][-1],sample_temperature)
            seq.append(index)
            if (index) == end_idx:
                break
        seq_list.append(seq)
    return seq_list


def sample(preds,sample_temperature):
    preds_ = np.log(preds).astype('float64')/sample_temperature
    probs= np.exp(preds_)/np.sum(np.exp(preds_))
    out=np.argmax(np.random.multinomial(1,probs,1))
    return out

def save_model():
    model.save_weights(path)

def load_model():
    model.load_weights(path)


nr_smiles = [100000]
for n in nr_smiles: 
    f_string = ''
    with open('./datasets/ChEMBL_filtered') as f:
        i = 0
        for line in f:
            if len(line)<98:
                f_string = f_string+line
                i+=1
            if(i>=n):
              break
    smiles = f_string.split('\n')[:-1]


emb_dim = 256
max_len = 100
epochs = 40
batch_size = 16
sample_temperature = 0.75
path = 'SeqRNN' + str(epochs)


vocabs,char_to_int,int_to_char = vocab(smiles)
vocab_size = len(vocabs)
smiles_tok = tokenize(smiles)


dataX =  encode(smiles_tok)
dataY = creat_label(dataX)

data_X = np.reshape(dataX, (n,max_len))  
data_Y = np.reshape(dataY, (n,max_len))       
x_shape = data_X[0].shape

model = generator_model()
model.summary()

model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy')

early_stop = EarlyStopping(monitor = "loss", patience=5)
checkpoint = ModelCheckpoint('detecting_parameter', monitor = 'loss', verbose = 1, mode = 'auto') 
callbacks_list = [checkpoint, early_stop]

results = model.fit(dataX, dataY, epochs = 120, batch_size =16, callbacks = callbacks_list)

fig, ax = plt.subplots()
ax.plot(results.history['loss'])
ax.set(xlabel='epochs', ylabel = 'loss')
figure_path = 'detecting_parameter' + "Loss_plot.png"
fig.savefig(figure_path)
plt.show()     



##########################分界线##########################

sample_temperature = 0.7
nums = 10000
generate_seq = generate_sample(char_to_int['G'], nums, char_to_int['A'],model)
generate_smiles = decode(generate_seq) 


from Measure import check_validity
from Measure import check_novelty
from Measure import check_uniqueness
from Measure import smi_mol
from Measure import draw_molecule

valid_molecules, valid_score = check_validity(generate_smiles)
novel_score = check_novelty(valid_molecules,smiles)
unique_molecules , unique_score = check_uniqueness(valid_molecules)
img=draw_molecule(valid_molecules)

print(valid_score)
print(novel_score)
print(unique_score)
img    
