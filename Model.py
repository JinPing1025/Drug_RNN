import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model


import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


class Generator():
    def __init__(self,vocab,vocab_size,emb_dim,max_len,epochs,batch_size):
        self.vocab = vocab
        
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.sample_temperature = 0.75
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, 
                                                     epsilon=1e-07, centered=False, name='RMSprop')
        self.path = 'SeqRNN' + str(epochs)
        
    def generator_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.vocab_size, self.emb_dim, input_length = self.max_len))
        self.model.add(LSTM(512, return_sequences=True))   
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(512, return_sequences=True))   
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.vocab_size, activation = 'softmax'))
        print(self.model.summary())
        self.model.compile(optimizer = self.optimizer, loss = 'sparse_categorical_crossentropy')
    
    def model_fit(self,dataX,dataY):
        
        early_stop = EarlyStopping(monitor = "loss", patience=5)
        checkpoint = ModelCheckpoint('detecting_parameter', monitor = 'loss', verbose = 1, mode = 'auto') 
        callbacks_list = [checkpoint, early_stop]
        
        results = self.model.fit(dataX, dataY, epochs = self.epochs, batch_size = self.batch_size, callbacks = callbacks_list)
        
        fig, ax = plt.subplots()
        ax.plot(results.history['loss'])
        ax.set(xlabel='epochs', ylabel = 'loss')
        figure_path = 'detecting_parameter' + "Loss_plot.png"
        fig.savefig(figure_path)
        plt.show()     
        last_epoch = early_stop.stopped_epoch
        return results, last_epoch
    
    def generate_sample(self,start_idx, num, end_idx):
        seq_list = []
        for j in tqdm(range(num)):
            seq = [start_idx]
            for i in range(self.max_len-1):
                x = np.reshape(seq, (1, len(seq),1))
                preds = self.model.predict(x, verbose=1)    #(1,1,33)
                index = self.sample(preds[0][-1],self.sample_temperature)
                seq.append(index)
                if (index) == end_idx:
                    break
            seq_list.append(seq)
        return seq_list
    
    def sample(self,preds,sample_temperature):
        preds_ = np.log(preds).astype('float64')/self.sample_temperature
        probs= np.exp(preds_)/np.sum(np.exp(preds_))
        out=np.argmax(np.random.multinomial(1,probs,1))
        return out
    
    def save_model(self):
        self.model.save_weights(self.path)
    
    def load_model(self):
        self.model.load_weights(self.path)



