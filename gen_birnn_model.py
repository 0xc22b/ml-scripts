#!/usr/bin/env python3
import os
import sys
import pickle
import warnings

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from keras.models import Sequential, model_from_json
from keras.layers.core import Activation, Dense, Dropout, Flatten, Merge
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.callbacks import BaseLogger, EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

class CustomModelCheckpoint(ModelCheckpoint):

    def __init__(self,
                 filepath,
                 monitor = 'val_loss',
                 verbose = 0,
                 save_best_only = False,
                 mode = 'auto',
                 stop_acc = None):

        super().__init__(filepath,
                         monitor = monitor,
                         verbose = verbose,
                         save_best_only = save_best_only,
                         mode = mode)

        self.stop_acc = stop_acc

    def on_epoch_begin(self, epoch, logs = {}):
        super().on_epoch_begin(epoch, logs = logs)

        self.seen = 0
        self.totals = {}

    def on_batch_end(self, batch, logs = {}):
        super().on_batch_end(batch, logs = logs)

        batch_size = logs.get('size', 0)
        self.seen += batch_size

        for k, v in logs.items():
            if k in self.totals:
                self.totals[k] += v * batch_size
            else:
                self.totals[k] = v * batch_size

    def on_epoch_end(self, epoch, logs = {}):

        custom_logs = {}
        for k, v in logs.items():
            custom_logs[k] = v

        if 'loss' in self.totals and 'loss' not in custom_logs:
            custom_logs['loss'] = self.totals['loss'] / self.seen
        if 'acc' in self.totals and 'acc' not in custom_logs:
            custom_logs['acc'] = self.totals['acc'] / self.seen

        super().on_epoch_end(epoch, custom_logs)

        if self.stop_acc:

            current = custom_logs.get('acc')
            if current is None:
                warnings.warn('Custom model checkpoint requires %s available!' % ('acc'),
                              RuntimeWarning)
            else:
                if current >= self.stop_acc:
                    if self.verbose > 0:
                        print('early stopping as acc meets')
                    self.model.stop_training = True

class BiRnnPredictor(object):

    def __init__(self, data_path, weight_path):

        self._data_path = data_path
        self._weight_path = weight_path

        self._words = []
        self._labels = []
        self._sentence_length = -1

        self._model = None

    def load(self):

        f = open(self._data_path, 'rb')
        data = pickle.load(f)

        self._words = data[0]
        self._labels = data[1]
        self._sentence_length = data[2]

        self._model = model_from_json(data[3])
        self._model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])

        f.close()

        self._model.load_weights(self._weight_path)

    def _save(self):

        f = open(self._data_path, 'wb')
        pickle.dump([self._words,
                     self._labels,
                     self._sentence_length,
                     self._model.to_json()],
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
        batch_size = 32
        nb_epoch = 100000

        model_checkpoint = CustomModelCheckpoint(filepath = self._weight_path,
                                                 monitor = 'acc',
                                                 verbose = 1,
                                                 save_best_only = True,
                                                 stop_acc = stop_acc)

        self._model.fit([X_train, X_train],
                        y_train,
                        batch_size = batch_size,
                        nb_epoch = nb_epoch,
                        verbose = 1,
                        shuffle = True,
                        callbacks = [model_checkpoint])

    def _build_model(self, word_length, label_length, sentence_length):

        embedding_size = 128
        lstm_size = 64
        dropout_rate = 0.1

        forward = Sequential()
        forward.add(Embedding(word_length,
                              embedding_size,
                              input_length = sentence_length))
        forward.add(LSTM(lstm_size, activation = 'tanh'))

        backward = Sequential()
        backward.add(Embedding(word_length,
                               embedding_size,
                               input_length = sentence_length))
        backward.add(LSTM(lstm_size, activation = 'tanh', go_backwards = True))

        model = Sequential()
        model.add(Merge([forward, backward], mode='concat'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(label_length, activation = 'softmax'))

        model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])

        self._model = model

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

        results = self._model.predict_classes([X_test, X_test], batch_size, verbose)

        label_results = []
        for result in results:
            label_results.append(self._labels[result])

        return label_results

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

    predictor = BiRnnPredictor(predictor_data_path, weight_path)
    predictor.fit(texts, labels)
