# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import theano
#import theano.tensor as T
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation,Flatten
#from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D, Input, ZeroPadding1D, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM,SimpleRNN
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam, SGD
import pickle
import sys

nb_filter = 64
filter_length = 7
input_dim = 20
embedding_size = 20
input_length = None
padding = filter_length // 2

AAs = ['X','I','L','V','F','M','C','A','G','P','T','S','Y','W','Q','N','H','E','D','K','R']
AAIndexes = {AAs[i] : i for i in range(len(AAs))}

def piecewise_scaling_func(x):
    if x < -5:
        y = 0.0
    elif -5 <= x <= 5:
        y = 0.5 + 0.1*x
    else:
        y = 1.0
    return y


    
class ReadData(object):

    @staticmethod
    def encodeAA(residue) :
    # Encode an amino acid sequence
    
        return [1 if residue == amino_acid else 0
            for amino_acid in ('A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 
                               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V')]
        if x not in AAIndexes :
            return 0
        return AAIndexes[x]
    

    @staticmethod
    def encode_dssp(dssp):
        return [1 if dssp == hec else 0 
                for hec in ('H', 'E', 'C')]
        y = 0
        for hec in ('H', 'E', 'C'):
            if dssp == hec:
                break;
            else:
                y += 1
            if y >= 2:
                break;
        return y

    @staticmethod
    def load(filename, window_size=19):
        print('... loading data ("%s")' % filename)
        n=1
        
        index = 0
        with open(filename, 'r') as f:
            line = f.read().strip().split('\n')
            num_proteins = len(line) // 2
            X = [None] * num_proteins
            Y = [None] * num_proteins
            for line_num in range(num_proteins):
                sequence = line[line_num*2]                                                     #Get the amino acid sequence on line line_num*2
                structure = line[line_num*2 + 1]                                                #Get the corresponding secondary structure from line_num*2 +1
                if len(sequence)!=len(structure):
                    print(line_num*2)                                               #Check if there is a protein has different number of AA and structure
                #if len(sequence)<maxlen:
                #    for i in range(len(sequence),maxlen):
                #        sequence += '0'
                #        structure += '0'
                
                X[line_num] = [ReadData.encodeAA(residue) for residue in sequence]
                #if line_num==1:
                #    print(X[line_num])
                Y[line_num] = np.array([ReadData.encode_dssp(dssp) for dssp in structure])
                index += len(sequence)
            #X = np.array(X)
            #print (X.shape)
            #open('X.txt','w').write(str(X))
        
        return X,Y,index

    @staticmethod
    def load_pssm(filename, scale=piecewise_scaling_func):
        print('... loading pssm ("%s")' % filename)

        index = 0
        with open(filename, 'r') as f:
            num_proteins = int(f.readline().strip())
            X = [None] * num_proteins
            Y = [None] * num_proteins
            index = 0 #[None] * num_proteins
            for protein_num in range(num_proteins):
                m = int(f.readline().strip())
                sequences = [None] * m
                for line_num in range(m):
                    line = f.readline()
                    sequences[line_num]=[]
                    for i in range(20):
                        s = ''.join(line[i*3: i*3+3]).strip()
                        sequences[line_num].append(scale(float(s)))
                    
                #double_end = ([0.]*20) * (window_size//2)
                #sequences = double_end + sequences + double_end
                X[protein_num] = sequences
                
                structure = f.readline().strip()
                Y[protein_num] = [ReadData.encode_dssp(dssp) for dssp in structure]

                index += m
        #X[0] = np.array(X[0])
        #print(X[0].shape)
        return X, Y, num_proteins,index
    
def get_batch(data_in, data_out):
    while True:
        for i in range(len(data_in)):
            input_data = np.array([data_in[i]])
            output_data = np.array([data_out[i]])
            #input_data = np.squeeze(input_data)
            #output_data = np.squeeze(output_data)
            yield (input_data, output_data)
    
    lengths = list(set(map(len, data_in)))
    data_by_length = {}
    for i, l in enumerate(map(len, data_in)):
        if l not in data_by_length:
            data_by_length[l] = []
        data_by_length[l].append(i)

    while True: # a new epoch
        np.random.shuffle(lengths)
        for length in lengths:
            indexes = data_by_length[length]
            np.random.shuffle(indexes)
            input_data = np.array([data_in[i] for i in indexes])
            output_data = np.array([data_out[i] for i in indexes])
            yield (input_data, output_data)
            
def get_Xbatch(data_in):
    lengths = list(set(map(len, data_in)))
    data_by_length = {}
    for i, l in enumerate(map(len, data_in)):
        if l not in data_by_length:
            data_by_length[l] = []
        data_by_length[l].append(i)

    while True: # a new epoch
        np.random.shuffle(lengths)
        for length in lengths:
            indexes = data_by_length[length]
            np.random.shuffle(indexes)
            input_data = np.array([data_in[i] for i in indexes])
            
            yield (input_data)

def get_model():
    model = Sequential()
    
    #We start with a embedding layer which convert our sequence to dense vectors
    #model.add(Embedding(2,embedding_size))
    
    #model.add(ZeroPadding1D(padding))
    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size filter_length:
    model.add(Convolution1D(nb_filter=64,
                                filter_length=20,
                                input_shape=(None,20),
                                border_mode='same',
                                activation='relu',
                                subsample_length = 1))
    model.add(Dropout(0.3)) 
    model.add(Convolution1D(nb_filter=64,
                                filter_length=20,
                                border_mode='same',
                                activation='relu',
                                subsample_length = 1))
    #model.add(GlobalMaxPooling1D())                             
    model.add(Dropout(0.3)) 
   
    model.add(Convolution1D(nb_filter=128,
                            filter_length=10,
                            border_mode='same',
                            activation='relu'))
    
    #model.add(Convolution1D(nb_filter=256,
    #                        filter_length=5,
    #                        border_mode='same',
    #                        activation='relu'))
    
    #model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(3))
    model.add(Activation('softmax'))
    return model
def main():
    
    X_train, Y_train, num_proteins = ReadData.load('training.data')
    X_valid, Y_valid, num_validproteins = ReadData.load('validation.data')
    X_eval, Y_eval, num_evalproteins = ReadData.load('casp9.data')
    
    model = get_model()
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    #Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.summary()
    #nb_samples = X_train[0].shape
    
    #test_size = int(0.1 * num_proteins)
    #X_test, Y_test =  X_train[-test_size:], Y_train[-test_size:]
    #print(Y_train[0])
    #print('\n\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape, model.output_shape, nb_filter, filter_length))
    
    fit_results = model.fit_generator(get_batch(X_train, Y_train),
                        samples_per_epoch=num_proteins,
                        validation_data=get_batch(X_valid, Y_valid),
                        nb_val_samples=num_validproteins,
                        nb_epoch=5)
        
    eval = model.evaluate_generator(get_batch(X_eval,Y_eval),
                                    num_evalproteins)
    print(eval)
    #print('valid accuracy:', acc,'valid loss',score)
    sys.setrecursionlimit(10000)
    pickle.dump(fit_results, open('CNN1D_fit_results.pkl', 'w'))
    model.save_weights('CNN1D_model_weights.hdf5')
    json_string = model.to_json()                                   #等价于 json_string = model.get_config()  
    open('my_model_architecture.json','w').write(json_string)    
        
    #加载模型数据和weights  
    #model = model_from_json(open('my_model_architecture.json').read())    
    #model.load_weights('CNN1D_model_weights.hdf5')
if __name__ == '__main__':
    main()
    
