#!/usr/bin/env python3
import sys
import os
import json
import pickle
import random
from datetime import datetime
from time import time
from collections import defaultdict

import sqlite3

import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold

from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.pipeline import Pipeline
from sklearn.grid_search import ParameterGrid

from sklearn import metrics

def _get_preprocessor_fpath(model_dpath):
    return os.path.join(model_dpath, 'preprocessor.pkl')

def _get_db_fpath(model_dpath):
    return os.path.join(model_dpath, 'ensemble_models.sqlite')

def _get_data_fpath(model_dpath, dir_name, base_name, model_id, fold_id = None):

    fname = base_name + '_' + _id_to_str(model_id)
    if fold_id != None:
        fname += '_' + _id_to_str(fold_id)
    fname += '.pkl'

    return os.path.join(model_dpath, 'data', dir_name, fname)

def _id_to_str(row_id):
    if row_id < 10:
        return '0' + str(row_id)

    return str(row_id)

def read_data(text_src):

    norm_texts = []
    semantic_tags = []

    text_src = open(text_src)
    for line in text_src:
        arr = line.strip().split('\t')

        if len(arr) < 2: continue

        norm_text = arr.pop(0)
        semantic_tag = arr.pop(0)

        norm_texts.append(norm_text)
        semantic_tags.append(semantic_tag)

    text_src.close()

    return norm_texts, semantic_tags

def process_data(model_dpath, X, y, n_folds):

    preprocessor_fpath = _get_preprocessor_fpath(model_dpath)
    if os.path.exists(preprocessor_fpath):
        f = open(preprocessor_fpath, 'rb')
        data = pickle.load(f)
        f.close()

        vectorizer = data[0]
        X = vectorizer.transform(X)

        label_encoder = data[1]
        y = label_encoder.transform(y)
    else:
        vectorizer = TfidfVectorizer(tokenizer=str.split, ngram_range=(1,3))
        X = vectorizer.fit_transform(X)

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        f = open(preprocessor_fpath, 'wb')
        pickle.dump([vectorizer, label_encoder], f, -1)
        f.close()

    if n_folds <= 1:
        return X, y, None

    folds = StratifiedKFold(y, n_folds)
    return X, y, folds

def init_db(model_dpath):

    db_fpath = _get_db_fpath(model_dpath)
    if os.path.exists(db_fpath):
        print("Found existing db! Use it.")
        return

    os.makedirs(model_dpath)
    os.makedirs(os.path.join(model_dpath, 'data'))
    os.makedirs(os.path.join(model_dpath, 'data', 'models'))
    os.makedirs(os.path.join(model_dpath, 'data', 'fitted_models'))
    os.makedirs(os.path.join(model_dpath, 'data', 'model_scores'))
    print("Created directories")

    create_table_query = """
        create table models (
            model_id       integer PRIMARY KEY NOT NULL,
            desc           text NOT NULL,
            created_at     timestamp NOT NULL
        );

        create table fitted_models (
            model_id       integer NOT NULL,
            fold_id        integer NOT NULL,
            score          real NOT NULL,
            created_at     timestamp NOT NULL
        );

        create table model_scores (
            model_id       integer UNIQUE NOT NULL,
            score          real NOT NULL,
            created_at     timestamp NOT NULL
        );

        create table ensemble (
            model_id       integer NOT NULL,
            weight         integer NOT NULL,
            created_at     timestamp NOT NULL
        );
    """

    create_index_query = """
        create index fitted_models_index
            on fitted_models (model_id, fold_id)
    """

    conn = sqlite3.connect(db_fpath)
    with conn:
        conn.executescript(create_table_query)
        conn.execute(create_index_query)
    conn.close()

    print("Created a new db")

def create_model_instances(config):

    print('Creating model instances...')

    models = []

    model_classess = {
        'ridge': RidgeClassifier,
        'perceptron': Perceptron,
        'passive_aggressive': PassiveAggressiveClassifier,
        'sgd': SGDClassifier,
        'nearest_centroid': NearestCentroid,
        'multinomial_nb': MultinomialNB,
        'linear_svc': LinearSVC,
        'svc': SVC,
        'dtree': DecisionTreeClassifier,
        'forest': RandomForestClassifier,
        'gbc': GradientBoostingClassifier,
        'extra': ExtraTreesClassifier
    }

    for model_name, param_grid in config.items():

        model_class = model_classess[model_name]
        print('    Building %s models' % str(model_class).split('.')[-1][:-2])

        models.extend([model_class(**p) for p in ParameterGrid(param_grid)])

    print('Created ' + str(len(models)) + ' model instances.')

    return models

def fetch_model_instances(model_dpath):

    print('Fetching model instances from db...')

    model_ids = []

    db_fpath = _get_db_fpath(model_dpath)
    conn = sqlite3.connect(db_fpath)

    cursor = conn.execute('select model_id from models order by model_id')
    for row in cursor:
        model_ids.append(row[0])

    conn.close()

    models = {}
    params = []

    for model_id in model_ids:
        data_fpath = _get_data_fpath(model_dpath, 'models', 'model', model_id)
        f = open(data_fpath, 'rb')
        model = pickle.load(f)
        f.close()

        models[model_id] = model

        if model.get_params() in params:
            raise ValueError('Duplicate model parameters!')
        params.append(model.get_params())

    print('Fetched ' + str(len(models)) + ' model instances')

    return models

def fetch_model_instance(model_dpath, model_id):

    model_instance = None

    data_fpath = _get_data_fpath(model_dpath, 'models', 'model', model_id)
    f = open(data_fpath, 'rb')
    model_instance = pickle.load(f)
    f.close()

    return model_instance

def save_model_instance(model_dpath, model_instance):

    db_fpath = _get_db_fpath(model_dpath)
    conn = sqlite3.connect(db_fpath)
    with conn:
        cursor = conn.cursor()
        cursor.execute('insert into models (desc, created_at) values (?, ?)',
                       (str(model_instance), datetime.now()))

    last_row_id = cursor.lastrowid
    conn.close()

    data_fpath = _get_data_fpath(model_dpath, 'models', 'model', last_row_id)
    f = open(data_fpath, 'wb')
    pickle.dump(model_instance, f, -1)
    f.close()

    return last_row_id

def save_fitted_model(model_dpath, fitted_model_id, fold_id, fitted_model, score):

    db_fpath = _get_db_fpath(model_dpath)
    conn = sqlite3.connect(db_fpath)
    with conn:
        cursor = conn.cursor()
        cursor.execute('insert into fitted_models (model_id, fold_id, score, created_at) values (?, ?, ?, ?)',
                       (fitted_model_id, fold_id, score, datetime.now()))

    last_row_id = cursor.lastrowid
    conn.close()

    data_fpath = _get_data_fpath(model_dpath, 'fitted_models', 'fitted_model', fitted_model_id, fold_id)
    f = open(data_fpath, 'wb')
    pickle.dump(fitted_model, f, -1)
    f.close()

    return last_row_id

def save_model_score(model_dpath, model_id, score, preds, probs):

    db_fpath = _get_db_fpath(model_dpath)
    conn = sqlite3.connect(db_fpath)
    with conn:
        cursor = conn.cursor()
        cursor.execute('insert into model_scores (model_id, score, created_at) values (?, ?, ?)',
                       (model_id, score, datetime.now()))

    last_row_id = cursor.lastrowid
    conn.close()

    data_fpath = _get_data_fpath(model_dpath, 'model_scores', 'preds', model_id)
    f = open(data_fpath, 'wb')
    pickle.dump(preds, f, -1)
    f.close()

    if probs != None:
        data_fpath = _get_data_fpath(model_dpath, 'model_scores', 'probs', model_id)
        f = open(data_fpath, 'wb')
        pickle.dump(probs, f, -1)
        f.close()

def find_model_id(model_instance, db_model_instances):
    for db_model_id, db_model_instance in db_model_instances.items():
        if model_instance.get_params() == db_model_instance.get_params():
            return db_model_id
    return None

def fit_models(model_dpath, model_instances, X, y, X_test, y_test):

    db_model_instances = fetch_model_instances(model_dpath)

    print()
    print('Fitting models...')
    for model_instance in model_instances:
        print('-' * 80)
        print(model_instance)
        print()

        if find_model_id(model_instance, db_model_instances):
            print('No fitting as already exists in db')
            print()
            continue

        model_id = save_model_instance(model_dpath, model_instance)
        fitted_model = fetch_model_instance(model_dpath, model_id)

        t0 = time()
        fitted_model.fit(X, y)
        train_time = (time() - t0) / 60
        print("    train time: %0.3f mins" % train_time)

        t0 = time()
        pred = fitted_model.predict(X_test)
        test_time = (time() - t0) / 60
        print("    test time:  %0.3f mins" % test_time)

        preds = pred

        score = metrics.accuracy_score(y_test, pred)
        print("    accuracy:   %0.4f" % score)

        save_fitted_model(model_dpath, model_id, 0, fitted_model, score)
        print("    saved to db")

        probs = None
        if hasattr(fitted_model, 'predict_proba'):
            t0 = time()
            probs = fitted_model.predict_proba(X_test)
            test_time = (time() - t0) / 60
            print("    predict time:  %0.3f mins" % test_time)

        save_model_score(model_dpath, model_id, score, preds, probs)
        print("saved to db")
        print()

    print('-' * 80)

def fit_folded_models(model_dpath, model_instances, X, y, folds):

    db_model_instances = fetch_model_instances(model_dpath)

    n_examples = X.shape[0]
    n_classes = len(np.unique(y))

    print()
    print('Fitting models...')
    for model_instance in model_instances:
        print('-' * 80)
        print(model_instance)
        print()

        if find_model_id(model_instance, db_model_instances):
            print('No fitting as already exists in db')
            print()
            continue

        model_id = save_model_instance(model_dpath, model_instance)
        preds = np.zeros((n_examples, ))
        probs = np.zeros((n_examples, n_classes))

        for fold_id, fold in enumerate(folds):

            print('Fold id: ' + str(fold_id))
            train_indexes, validate_indexes = fold
            X_train, y_train = X[train_indexes], y[train_indexes]
            X_validate, y_validate = X[validate_indexes], y[validate_indexes]

            fitted_model = fetch_model_instance(model_dpath, model_id)

            t0 = time()
            fitted_model.fit(X_train, y_train)
            train_time = time() - t0
            print("    train time: %0.3fs" % train_time)

            t0 = time()
            pred = fitted_model.predict(X_validate)
            test_time = time() - t0
            print("    test time:  %0.3fs" % test_time)

            preds[validate_indexes] = pred

            score = metrics.accuracy_score(y_validate, pred)
            print("    accuracy:   %0.3f" % score)

            save_fitted_model(model_dpath, model_id, fold_id, fitted_model, score)
            print("    saved to db")

            if hasattr(fitted_model, 'predict_proba'):
                t0 = time()
                probs[validate_indexes] = fitted_model.predict_proba(X_validate)
                test_time = time() - t0
                print("    predict time:  %0.3fs" % test_time)

        if hasattr(fitted_model, 'predict_proba'):
            score = metrics.accuracy_score(y, np.argmax(probs, axis=1))
        else:
            score = metrics.accuracy_score(y, preds)
            probs = None

        save_model_score(model_dpath, model_id, score, preds, probs)
        print("saved to db")
        print()

    print('-' * 80)

def main():

    config_fpath = sys.argv[1]
    model_dpath = sys.argv[2]
    train_fpath = sys.argv[3]
    test_fpath = None

    n_folds = 1
    if n_folds <= 1:
        test_fpath = sys.argv[4]

    with open(config_fpath, encoding='utf-8') as config_file:
        config = json.loads(config_file.read())
    print("Found configuration")

    init_db(model_dpath)

    print("Loading data...")
    X, y = read_data(train_fpath)
    X, y, folds = process_data(model_dpath, X, y, n_folds)

    if test_fpath:
        X_test, y_test = read_data(test_fpath)
        X_test, y_test, _ = process_data(model_dpath, X_test, y_test, 1)
    print("Loaded and transformed data")

    model_instances = create_model_instances(config)
    if n_folds > 1:
        fit_folded_models(model_dpath, model_instances, X, y, folds)
    else:
        fit_models(model_dpath, model_instances, X, y, X_test, y_test)

if __name__ == '__main__':
    main()
