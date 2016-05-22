#!/usr/bin/env python3
import sys
import os
import json
import pickle
import random
from datetime import datetime
from time import time
from collections import defaultdict
from collections import Counter

import sqlite3

import numpy as np

from sklearn import metrics

from gen_ensemble_models import read_data, process_data
from gen_ensemble_models import _get_preprocessor_fpath, _get_db_fpath, _get_data_fpath
from gen_ensemble_models import create_model_instances
from gen_ensemble_models import fetch_model_instances
from gen_ensemble_models import find_model_id

from voting_classifier import VotingClassifier

def fetch_fitted_models(model_dpath, model_id):

    fitted_models = []

    db_fpath = _get_db_fpath(model_dpath)
    conn = sqlite3.connect(db_fpath)
    cursor = conn.execute('select model_id, fold_id from fitted_models where model_id = ? order by fold_id',
                          (model_id,))
    for row in cursor:

        data_fpath = _get_data_fpath(model_dpath, 'fitted_models', 'fitted_model', row[0], row[1])
        f = open(data_fpath, 'rb')
        fitted_model = pickle.load(f)
        f.close()

        fitted_models.append((row[0], row[1], fitted_model))

    conn.close()

    return fitted_models

def fetch_model_scores(model_dpath, voting, max_size):

    print('Fetching model scores from db...')

    query = 'select model_id, score from model_scores order by score desc'

    scores = []

    db_fpath = _get_db_fpath(model_dpath)
    conn = sqlite3.connect(db_fpath)
    cursor = conn.execute(query)
    for row in cursor:
        scores.append((row[0], row[1]))
    conn.close()

    if voting == 'soft':
        db_scores = scores
        scores = []
        for model_id, score in db_scores:
            data_fpath = _get_data_fpath(model_dpath, 'model_scores', 'probs', model_id)
            if os.path.exists(data_fpath):
                scores.append((model_id, score))

    print('Fetched ' + str(len(scores)) + ' model scores')
    print('    ids: ' + ', '.join([str(score[0]) for score in scores]))

    if len(scores) > max_size:
        scores = scores[:max_size]

    return scores

def fetch_model_preds(model_dpath, model_id):

    preds = None

    data_fpath = _get_data_fpath(model_dpath, 'model_scores', 'preds', model_id)
    f = open(data_fpath, 'rb')
    preds = pickle.load(f)
    f.close()

    return preds

def fetch_model_probs(model_dpath, model_id):

    probs = None

    data_fpath = _get_data_fpath(model_dpath, 'model_scores', 'probs', model_id)
    f = open(data_fpath, 'rb')
    probs = pickle.load(f)
    f.close()

    return probs

def save_ensemble(model_dpath, ensemble_fpath, voting, ensemble):

    values = ((model_id, weight, datetime.now()) for model_id, weight in ensemble.most_common())

    db_fpath = _get_db_fpath(model_dpath)
    conn = sqlite3.connect(db_fpath)
    with conn:
        conn.execute('delete from ensemble')
        conn.executemany('insert into ensemble (model_id, weight, created_at) values (?, ?, ?)',
                         values)
    conn.close()
    print('saved ensemble to db')

    classifiers = []
    for model_id, weight in ensemble.items():
        fitted_models = fetch_fitted_models(model_dpath, model_id)
        if len(fitted_models) == 1:
            classifiers.append(fitted_models[0][2])
        else:
            models = [fm[2] for fm in fitted_models]
            classifiers.append(VotingClassifier(models, voting=voting))

    voting_classifier = VotingClassifier(classifiers, voting=voting)

    f = open(ensemble_fpath, 'wb')
    pickle.dump(voting_classifier, f, -1)
    f.close()
    print('saved voting classifier to ' + ensemble_fpath)

def get_ensemble_score(model_dpath, voting, ensemble, X, y):

    if voting == 'soft':
        return _get_soft_voting_score(model_dpath, ensemble, X, y)
    else:
        return _get_hard_voting_score(model_dpath, ensemble, X, y)

def _get_soft_voting_score(model_dpath, ensemble, X, y):

    n_examples = X.shape[0]
    n_classes = len(np.unique(y))

    ensemble_weight = 0.0

    ensemble_probs = np.zeros((n_examples, n_classes))
    for model_id, weight in ensemble.items():
        probs = fetch_model_probs(model_dpath, model_id)
        ensemble_probs += probs * weight

        ensemble_weight += weight

    ensemble_probs /= ensemble_weight
    ensemble_score = metrics.accuracy_score(y, np.argmax(ensemble_probs, axis=1))

    return ensemble_score, ensemble_probs, ensemble_weight

def _get_hard_voting_score(model_dpath, ensemble, X, y):

    ensemble_preds = []
    ensemble_weights = []

    ensemble_weight = 0.0

    ensemble_probs = {}
    for model_id, weight in ensemble.items():
        preds = fetch_model_preds(model_dpath, model_id)
        ensemble_probs[model_id] = {
            'preds': preds,
            'weight': weight
        }

        ensemble_weight += weight

        ensemble_preds.append(preds)
        ensemble_weights.append(weight)

    y_preds = np.asarray(ensemble_preds, dtype=int).T
    y_preds = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=ensemble_weights)),
                                  axis=1,
                                  arr=y_preds)
    ensemble_score = metrics.accuracy_score(y, y_preds)

    return ensemble_score, ensemble_probs, ensemble_weight

def get_added_ensemble_score(model_dpath, voting, ensemble_probs, ensemble_weight, y, added_model_id):

    if voting == 'soft':
        return _get_added_soft_voting_score(model_dpath, ensemble_probs, ensemble_weight, y, added_model_id)
    else:
        return _get_added_hard_voting_score(model_dpath, ensemble_probs, ensemble_weight, y, added_model_id)

def _get_added_soft_voting_score(model_dpath, ensemble_probs, ensemble_weight, y, added_model_id):

    added_weight = ensemble_weight + 1.0

    added_probs = fetch_model_probs(model_dpath, added_model_id)
    added_probs = (ensemble_probs * ensemble_weight + added_probs) / added_weight

    added_score = metrics.accuracy_score(y, np.argmax(added_probs, axis=1))

    return added_score, added_probs, added_weight

def _get_added_hard_voting_score(model_dpath, ensemble_probs, ensemble_weight, y, added_model_id):

    added_preds = []
    added_weights = []

    added_weight = ensemble_weight + 1.0

    added_probs = ensemble_probs.copy()
    if added_model_id in added_probs:
        added_probs[added_model_id]['weight'] += 1.0
    else:
        preds = fetch_model_preds(model_dpath, added_model_id)
        added_probs[added_model_id] = {
            'preds': preds,
            'weight': 1.0
        }

    for model_id, values in added_probs.items():
        added_preds.append(values['preds'])
        added_weights.append(values['weight'])

    y_preds = np.asarray(added_preds, dtype=int).T
    y_preds = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=added_weights)),
                                  axis=1,
                                  arr=y_preds)
    added_score = metrics.accuracy_score(y, y_preds)

    return added_score, added_probs, added_weight

def build_ensemble_from_candidates(model_dpath, voting, candidate_model_ids, X, y):

    use_epsilon = False
    epsilon = 0.01

    init_model_size = 1
    max_model_size = 5

    ensemble = Counter(candidate_model_ids[:init_model_size])
    score, probs, model_size = get_ensemble_score(model_dpath, voting, ensemble, X, y)
    print('    Acc: ' + str(score) + ' from ' + str(model_size) + ' models')

    candidate_ensembles = [{'ensemble': Counter(ensemble), 'score': score}]
    while model_size < max_model_size:

        added_scores = []
        for candidate_model_id in candidate_model_ids:
            added_score, _, _ = get_added_ensemble_score(model_dpath,
                                                         voting,
                                                         probs,
                                                         model_size,
                                                         y,
                                                         candidate_model_id)

            added_scores.append({'model_id': candidate_model_id, 'score': added_score})

        added_scores = sorted(added_scores, key=lambda x: x['score'], reverse=True)
        added_score = added_scores[0]['score']

        if use_epsilon:
            diff_score = added_score - score
            if diff_score < epsilon:
                print('    New acc: ' + str(added_score) + ' improved less than epsilon. Stopped.')
                break

        added_model_id = added_scores[0]['model_id']
        ensemble.update({added_model_id: 1})

        score, probs, model_size = get_added_ensemble_score(model_dpath,
                                                            voting,
                                                            probs,
                                                            model_size,
                                                            y,
                                                            added_model_id)
        print('    Acc: ' + str(score) + ' from ' + str(int(model_size)) + ' models')

        if not use_epsilon:
            candidate_ensembles.append({'ensemble': Counter(ensemble), 'score': score})

    if not use_epsilon:
        candidate_ensembles = sorted(candidate_ensembles, key=lambda x: x['score'], reverse=True)
        ensemble = candidate_ensembles[0]['ensemble']
        print('    Picked the best ensemble: ' + str(ensemble))

    return ensemble

def build_ensemble(model_dpath, ensemble_fpath, voting, model_ids, X, y):

    print()
    print('Building ensemble...')

    candidate_model_size = len(model_ids)
    candidate_model_size_each_round = len(model_ids)
    n_rounds = 1
    print(str(candidate_model_size) + ' candidates')
    print(str(candidate_model_size_each_round) + ' candidates per round')
    print(str(n_rounds) + ' rounds')

    n_examples = X.shape[0]
    n_classes = len(np.unique(y))

    rs = np.random.mtrand._rand

    ranked_model_scores = fetch_model_scores(model_dpath, voting, candidate_model_size)
    ensemble = Counter()

    for i in range(n_rounds):
        print('-' * 80)
        print("Round " + str(i))

        random_indexes = rs.permutation(len(ranked_model_scores))[:candidate_model_size_each_round]
        candidate_model_ids = [ranked_model_scores[i][0] for i in sorted(random_indexes)]
        print('    Candidate model ids: ' + ', '.join(str(x) for x in candidate_model_ids))

        sub_ensemble = build_ensemble_from_candidates(model_dpath, voting, candidate_model_ids, X, y)

        ensemble += sub_ensemble

    print()
    print('Final ensemble: ' + str(ensemble))
    return ensemble

def main():

    config_fpath = sys.argv[1]
    model_dpath = sys.argv[2]
    train_fpath = sys.argv[3]
    ensemble_fpath = sys.argv[4]

    voting = 'hard'

    with open(config_fpath, encoding='utf-8') as config_file:
        config = json.loads(config_file.read())
    print("Found configuration")

    print("Loading data...")
    X, y = read_data(train_fpath)
    X, y, _ = process_data(model_dpath, X, y, 1)
    print("Loaded and transformed data")

    model_instances = create_model_instances(config)
    db_model_instances = fetch_model_instances(model_dpath)

    print('Checking model instances...')
    model_ids = []
    for model_instance in model_instances:
        model_id = find_model_id(model_instance, db_model_instances)
        if not model_id:
            raise ValueError('Found missing model: %s' % model_instance)
        model_ids.append(model_id)
    print('All model instances are available')

    ensemble = build_ensemble(model_dpath, ensemble_fpath, voting, model_ids, X, y)

    save_ensemble(model_dpath, ensemble_fpath, voting, ensemble)

if __name__ == '__main__':
    main()
