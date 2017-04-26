# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import theano
#import theano.tensor as T

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, ZeroPadding2D, Flatten
from keras.layers import Convolution2D, GlobalMaxPooling2D, MaxPooling2D

batch_size = 20
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 2

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

        X = []
        Y = []
        index = [0]
        with open(filename, 'r') as f:
            line = f.read().strip().split('\n')
            num_proteins = len(line) // 2

            for line_num in range(num_proteins):
                sequence = line[line_num*2]
                structure = line[line_num*2 + 1]

                double_end = [None] * (window_size // 2)
                unary_sequence = []
                for residue in double_end + list(sequence) + double_end:
                    unary_sequence += ReadData.encode_residue(residue)

                X += [
                    unary_sequence[start: start+window_size*20]
                    for start in range(0, len(sequence)*20, 20)
                ]

                Y += [ReadData.encode_dssp(dssp) for dssp in structure]

                index.append(index[-1] + len(sequence))

        return ReadData.shared_dataset([X, Y, index])

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
            input_data = np.expand_dims(input_data, axis=0)
            output_data = np.expand_dims(output_data, axis=0)
            yield (input_data, output_data)
    
def main():
    
    X_train, Y_train, protein_train, index_train = ReadData.load_pssm('train.pssm')
    X_test, Y_test, protein_test, index_test = ReadData.load_pssm('valid.pssm')
    model = Sequential()
    #model.add(Embedding())
    #model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(None,None,20) ,activation='relu'))
    model.add(Dropout(0.5))
    #model.add(MaxPooling2D())
    model.add(Convolution2D(128, 3, 3, border_mode='same' ,activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Flatten())
    model.add(Convolution2D(256, 3, 3, border_mode='same' ,activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(32,input_shape=(20,),activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    fit_result = model.fit_generator(get_batch(X_train, Y_train), 
                samples_per_epoch=index_train, 
                validation_data=get_batch(X_test, Y_test),
                nb_val_samples=index_test,
                nb_epoch=10)
    
    
    #model.fit(X_train,Y_train,
    #            batch_size=20,
    #            nb_epoch=10,
    #            validation_data=(X_test,Y_test),
    #            verbose=1)
    sys.setrecursionlimit(10000)
    pickle.dump(fit_results, open('CNN2D_fit_results.pkl', 'w'))                #save results
    model.save_weights('CNN2D_model_weights.hdf5')                              #save model weights
    json_string = model.to_json()                                               #save model to json file
    open('CNN2D_architecture.json','w').write(json_string)
    
    score = model.evaluate_generator(get_batch(X_test,Y_test),index_test)
    
    print(score)

if __name__ == '__main__':
    main()
