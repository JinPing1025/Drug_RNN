
import numpy as np

from vocabulary import Vocabulary
from Model import Generator

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
sample_temperature = 0.7
path = 'SeqRNN' + str(epochs)

vocab = Vocabulary()
vocab.vocab(smiles)
smiles_tok = vocab.tokenize(smiles)
vocabs,char_to_int,int_to_char = vocab.vocab(smiles)
vocab_size = len(vocabs)

dataX = vocab.encode(smiles_tok)
dataY = vocab.creat_label(dataX)

data_X = np.reshape(dataX, (n,max_len))  
data_Y = np.reshape(dataY, (n,max_len))       
x_shape = data_X[0].shape


model = Generator(vocab,vocab_size,emb_dim=256,max_len=100,epochs=40,batch_size = 16,x_shape=x_shape)
generate_model = model.generator_model()
generate_model.summary()

results, last_epoch = model.model_fit(data_X, data_Y)

nums = 100
generate_seq = model.generate_sample(char_to_int['G'], nums, char_to_int['A'])
generate_smiles = vocab.decode(generate_seq)


##########################分界线##########################

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
