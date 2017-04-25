# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import theano
#import theano.tensor as T

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten
#from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D, Input

batch_size = 19
nb_filter = 250
filter_length = 19
hidden_dims = 250
nb_epoch = 50
maxlen=470
def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

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
    def encode_residue(residue):
        return [1 if residue == amino_acid else 0
                for amino_acid in ('A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H',
                                   'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
                                   'Y', 'V')]

    @staticmethod
    def encode_dssp(dssp):
        return [1 if dssp == hec else 0 for hec in ('H', 'E', 'C')]

    @staticmethod
    def shared_dataset(data_xy, borrow=True):
        data_x, data_y, index = data_xy
        shared_x = theano.shared(floatX(data_x), borrow=borrow)
        shared_y = theano.shared(floatX(data_y), borrow=borrow)
        return shared_x, shared_y, index

    @staticmethod
    def load(filename, window_size=19):
        print('... loading data ("%s")' % filename)
        n=1
        
        index = [0]
        with open(filename, 'r') as f:
            line = f.read().strip().split('\n')
            num_proteins = len(line) // 2
            X = [None] * num_proteins
            Y = [None] * num_proteins
            for line_num in range(num_proteins):
                sequence = line[line_num*2]                                                     #Get the amino acid sequence on line line_num*2
                structure = line[line_num*2 + 1]                                                #Get the corresponding secondary structure from line_num*2 +1
                #double_end = [None] * (window_size // 2)
                if len(sequence)<maxlen:
                    for i in range(len(sequence),maxlen):
                        sequence += '0'
                        structure += '0'
                unary_sequence = []
                unary_sequence += [ReadData.encode_residue(residue) for residue in sequence]
                #X = np.array(X)
                #if n==1:
                #    print (unary_sequence)
                #    n = 0
                #for residue in double_end + list(sequence) + double_end:
                #    unary_sequence += ReadData.encode_residue(residue)
                #unary_sequence = np.asarray(unary_sequence)
                #print(len(unary_sequence))
                #unary_sequence = unary_sequence.tolist()
                X[line_num] = np.array([
                    unary_sequence[start:start+window_size]
                    for start in range(0, (maxlen-window_size))]
                )
                Y[line_num] = [ReadData.encode_dssp(dssp) for dssp in structure[:-window_size]]
                index.append(index[-1] + len(sequence))
            Y = np.array(Y)
            #print (X[0].shape)
            #X = X.tolist()
            #X = np.atleast_3d(np.asarray(X))
            print (Y.shape)
            #open('X.txt','w').write(str(X))
        #return ReadData.shared_dataset([X, Y, index])
        return X,Y,index

    @staticmethod
    def load_pssm(filename, window_size=19, scale=piecewise_scaling_func):
        print('... loading pssm ("%s")' % filename)

        X = []
        Y = []
        index = [0]
        with open(filename, 'r') as f:
            num_proteins = int(f.readline().strip())
            for __ in range(num_proteins):
                m = int(f.readline().strip())
                sequences = []
                for __ in range(m):
                    line = f.readline()
                    sequences += [scale(float(line[i*3: i*3+3]))
                                  for i in range(20)]

                double_end = ([0.]*20) * (window_size//2)
                sequences = double_end + sequences + double_end
                X += [
                    sequences[start:start+window_size*20]
                    for start in range(0, m*20, 20)
                ]

                structure = f.readline().strip()
                Y += [ReadData.encode_dssp(dssp) for dssp in structure]
                keras.preprocessing.sequence.pad_sequences(X)
                keras.preprocessing.sequence.pad_sequences(Y)
                index.append(index[-1] + m)

        return ReadData.shared_dataset([X, Y, index])
    
    
def main():
    
    X_train, Y_train, index_train = ReadData.load('train.data')
    X_valid, Y_valid, index_valid = ReadData.load('test.data')
    model = Sequential()
    input_length = None
    
    #sequence_text = Input(name='input_source', shape=(input_length,), dtype='int32')
    #embedded_text = Embedding(vocabulary_size, embedding_size, input_length=input_length, weights=[weights_w2v])(sequence_text)
    
    #cnn1d_text_pad = ZeroPadding1D(padding)(embedded_text)
    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size filter_length:
    model.add(Convolution1D(nb_filter=16,
                            filter_length=4,
                            input_shape=(19,20),
                            border_mode='valid',
                            activation='relu'
                            ))

	# we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    #model.add(Embedding(max_features,
    #                    embedding_dims,
     #                   input_length=maxlen,
     #                   dropout=0.2))
    
    # add a global max pooling:
    model.add(GlobalMaxPooling1D())
    
    # add a hidden layer:
    model.add(Dense(16))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    
    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()
    nb_samples = X_train[0].eval().shape[0]
    test_size = int(0.1 * nb_samples)
    X_test, Y_test =  X_train[0][-test_size:], Y_train[0][-test_size:]
    
    print('\n\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape, model.output_shape, nb_filter, filter_length))
    model.fit(X_train[0], Y_train[0],
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_split=0.2)
              #validation_data=(X_test, Y_test))
              
    eval = model.evaluate(X_test,Y_test)
    print (eval)
    #print('valid accuracy:', acc,'valid loss',score)
if __name__ == '__main__':
    main()
    