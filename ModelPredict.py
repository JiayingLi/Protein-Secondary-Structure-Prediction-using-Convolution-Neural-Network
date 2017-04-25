from keras.models import Model, model_from_json
import CNN1
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
def get_Xbatch(data_in):
    for i in range(len(data_in)):
        input_data = np.array([data_in[i]])
        
        yield (input_data)
    if False:
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
                #output_data = np.array([data_out[i] for i in indexes])
                yield (input_data)
            
#fit_pickle = 'CNN1D_fit_results.pkl'
#fit_results = pickle.load(open(fit_pickle))
#print(fit_results.history)

model = model_from_json(open('my_model_architecture.json').read())    
model.load_weights('CNN1D_model_weights.hdf5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
X_test, Y_test, num_proteins = CNN1.ReadData.load('train.data')
y_predict = model.predict_generator(get_Xbatch(X_test),num_proteins)
print(y_predict)

