#!/usr/bin/env python3
import os
import sys
import pickle

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from keras.models import Sequential, model_from_json
from keras.layers.core import Activation, Dense, Dropout, Flatten, Merge
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

from gen_birnn_model import CustomModelCheckpoint

class ConvPredictor(object):

    def __init__(self, data_path, weight_path):

        self._data_path = data_path
        self._weight_path = weight_path

        self._words = []
        self._labels = []
        self._sentence_length = -1

        self._model = None
        self._merge_size = -1

    def load(self):

        f = open(self._data_path, 'rb')
        data = pickle.load(f)

        self._words = data[0]
        self._labels = data[1]
        self._sentence_length = data[2]

        self._model = model_from_json(data[3])
        self._model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

        self._merge_size = data[4]

        f.close()

        self._model.load_weights(self._weight_path)

    def _save(self):

        f = open(self._data_path, 'wb')
        pickle.dump([self._words,
                     self._labels,
                     self._sentence_length,
                     self._model.to_json(),
                     self._merge_size],
                    f,
                    -1)
        f.close()
        print('saving model definition to ' + self._data_path)

    def fit(self, texts, labels):

        self._labels = list(set(labels))

        sentence_length = 0
        for text in texts:
            words = text.split(' ')
            if len(words) > sentence_length:
                sentence_length = len(words)
        self._sentence_length = sentence_length

        X_train = []
        y_train = []

        for text, label in zip(texts, labels):
            X_train.append(self._text_to_word_seq(text))
            y_train.append(self._labels.index(label))

        print(len(X_train), 'train sequences')

        self._build_model(len(self._words), len(self._labels), self._sentence_length)

        X_train = pad_sequences(X_train, maxlen = self._sentence_length)
        print('X_train shape:', X_train.shape)

        y_train = np_utils.to_categorical(np.array(y_train)).astype(np.bool)

        self._save()

        stop_acc = 0.9999
        batch_size = 50
        nb_epoch = 100000

        model_checkpoint = CustomModelCheckpoint(filepath = self._weight_path,
                                                 monitor = 'acc',
                                                 verbose = 1,
                                                 save_best_only = True,
                                                 stop_acc = stop_acc)

        if self._merge_size == 1:
            concat_X_train = X_train
        else:
            concat_X_train  = []
            for i in range(0, self._merge_size):
                concat_X_train.append(X_train)

        self._model.fit(concat_X_train,
                        y_train,
                        batch_size = batch_size,
                        nb_epoch = nb_epoch,
                        verbose = 1,
                        shuffle = True,
                        callbacks = [model_checkpoint])

    def _build_model(self, word_length, label_length, sentence_length):

        ngram_filters = [3, 4, 5]
        embedding_size = 300
        nb_filter = 100
        activation_func = 'relu'
        dropout_rate = 0.5

        model = Sequential()

        if len(ngram_filters) == 1:

            ngram_filter = ngram_filters[0]
            pool_length_ratio = sentence_length - ngram_filter + 1

            model.add(Embedding(word_length,
                                embedding_size,
                                input_length = sentence_length))
            model.add(Convolution1D(nb_filter,
                                    ngram_filter,
                                    activation = activation_func))
            model.add(MaxPooling1D(pool_length = pool_length_ratio))
            model.add(Flatten())

        else:
            conv_layers = []

            for ngram_filter in ngram_filters:

                pool_length_ratio = sentence_length - ngram_filter + 1

                sequential = Sequential()

                sequential.add(Embedding(word_length,
                                         embedding_size,
                                         input_length = sentence_length))
                sequential.add(Convolution1D(nb_filter,
                                             ngram_filter,
                                             activation = activation_func))
                sequential.add(MaxPooling1D(pool_length = pool_length_ratio))
                sequential.add(Flatten())

                conv_layers.append(sequential)

            model.add(Merge(conv_layers, mode = 'concat'))

        model.add(Dropout(dropout_rate))
        model.add(Dense(label_length, activation = 'softmax'))

        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

        self._model = model
        self._merge_size = len(ngram_filters)

    def _text_to_word_seq(self, text, do_fit = True):

        word_indices = []

        words = text.split(' ')
        for word in words:
            if word not in self._words:
                if not do_fit:
                    continue
                self._words.append(word)

            word_indices.append(self._words.index(word))

        return word_indices

    def predict(self, texts, batch_size = 64, verbose = 0):

        X_test = []

        for text in texts:
            X_test.append(self._text_to_word_seq(text, False))

        print(len(X_test), 'test sequences')

        X_test = pad_sequences(X_test, maxlen = self._sentence_length)
        print('X_test shape:', X_test.shape)

        if self._merge_size == 1:
            concat_X_test = X_test
        else:
            concat_X_test  = []
            for i in range(0, self._merge_size):
                concat_X_test.append(X_test)

        results = self._model.predict_classes(concat_X_test, batch_size, verbose)

        label_results = []
        for result in results:
            label_results.append(self._labels[result])

        return label_results

    def plot_model(self, to_file):

        plot(self._model, to_file)

if __name__ == '__main__':

    training_data_path = sys.argv[1]
    predictor_data_path = sys.argv[2]
    weight_path = sys.argv[3]

    texts = []
    labels = []

    training_data = open(training_data_path, 'r')
    for line in training_data:
        arr = line.strip().split('\t')
        text = arr[0]
        label = arr[1]

        texts.append(text)
        labels.append(label)

    predictor = ConvPredictor(predictor_data_path, weight_path)
    predictor.fit(texts, labels)
