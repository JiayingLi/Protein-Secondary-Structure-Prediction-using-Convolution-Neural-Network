# vocabulary = OrderedDict()
input_length = None
vocabulary_size = max(vocabulary.values()) + 1
weights_w2v = list(map(Word2Vec.__getitem__, vocabulary.keys()))
embedding_size = len(weights_w2v[0])
nb_classes = 5

# CNN hyperparms
nb_filter = 64
filter_length = 5
border_mode = 'valid'
padding = filter_length // 2
stride = 1

sequence_text = Input(name='input_source', shape=(input_length,), dtype='int32')
embedded_text = Embedding(vocabulary_size, embedding_size, input_length=input_length)(sequence_text)

cnn1d_text_pad = ZeroPadding1D(padding)(embedded_text)
cnn1d_text = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, 
            border_mode=border_mode, subsample_length=stride, activation='relu')(cnn1d_text_pad)

rnn_text = RNN(self.nb_hidden, return_sequences=True, activation='relu')(cnn1d_text)	
drop_text = Dropout(0.5)(rnn_text)
output_text = TimeDistributed(Dense(output_dim=nb_classes, activation='softmax'), name='output_source')(drop_text)

text_model = Model(input=[sequence_text], output=output_text)
text_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

def get_batch(data_in, data_out):
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
            input_data = {'input_source': np.array([data_in[i] for i in indexes])}
            output_data = {'output_source': np.array([data_out[i] for i in indexes])}
            yield (input_data, output_data)

text_model.fit_generator(get_batch(X_train, Y_train),
                        samples_per_epoch=len(X_train),
                        validation_data=get_batch(X_test, Y_train),
                        nb_val_samples=len(X_test),
                        nb_epoch=10)