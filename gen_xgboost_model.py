#!/usr/bin/env python3
import sys
import os
import pickle
import random

from pprint import pprint
from time import time
from collections import defaultdict

import numpy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn import svm

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV

from sklearn import metrics

import xgboost as xgb

class TreePredictor(object):

    def __init__(self, data_path, weight_path):

        self._data_path = data_path
        self._weight_path = weight_path

        self._classifier = None
        self._count_vect = None
        self._tfidf_transformer = None

        self._labels = []
        self._param = None

    def load(self):

        f = open(self._data_path, 'rb')
        data = pickle.load(f)

        self._count_vect = data[0]
        self._tfidf_transformer = data[1]
        self._labels = data[2]
        self._param = data[3]

        f.close()

        self._classifier = xgb.Booster(self._param)
        self._classifier.load_model(self._weight_path)

    def _save(self):

        f = open(self._data_path, 'wb')
        pickle.dump([self._count_vect,
                     self._tfidf_transformer,
                     self._labels,
                     self._param],
                    f,
                    -1)
        f.close()

        print('saving model definition to ' + self._data_path)
        self._classifier.save_model(self._weight_path)

        base_name = os.path.splitext(self._weight_path)[0]
        self._classifier.dump_model(base_name + '_raw.txt')

    def fit(self, texts, labels):

        self._labels = list(set(labels))

        y_train = []
        for label in labels:
            y_train.append(self._labels.index(label))

        self._count_vect = CountVectorizer(tokenizer=str.split, ngram_range=(1,3))
        self._tfidf_transformer = TfidfTransformer()

        X_train = self._count_vect.fit_transform(texts)
        X_train = self._tfidf_transformer.fit_transform(X_train)

        matrix_train = xgb.DMatrix(X_train, label = y_train)

        self._param = {
            'objective':'multi:softmax',
            'eta': 0.05,
            'max_depth': 7,
            'nthread': 8,
            'num_class': len(self._labels),
            'sub_sample': 0.9
        }

        num_round = 200
        watchlist  = [(matrix_train, 'train')]

        self._classifier = xgb.train(self._param,
                                     matrix_train,
                                     num_round,
                                     watchlist,
                                     early_stopping_rounds = 3)
        self._save()

    def predict(self, texts):

        selected_words = []

        X_test = self._count_vect.transform(texts)
        X_test = self._tfidf_transformer.transform(X_test)

        matrix_test = xgb.DMatrix(X_test)

        results = self._classifier.predict(matrix_test);

        label_results = []
        for result in results:
            label_results.append(self._labels[int(result)])

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

    predictor = TreePredictor(predictor_data_path, weight_path)
    predictor.fit(texts, labels)
