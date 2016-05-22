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

if __name__ == '__main__':

    text_src = sys.argv[1]
    output = sys.argv[2]
    model_output = sys.argv[3]

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

    lines = []
    accurate_count = 0

    count_vect = CountVectorizer(tokenizer=str.split, ngram_range=(1,3))
    tfidf_transformer = TfidfTransformer()
    clf = svm.LinearSVC()

    pipeline_clf = Pipeline([
        ('vect', count_vect),
        ('tfidf', tfidf_transformer),
        ('clf', clf)
    ])

    parameters = {'clf__C': (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                             1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                             2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
                             3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0)}
    gs_clf = GridSearchCV(pipeline_clf, parameters, n_jobs=-1, verbose=1)

    t0 = time()
    gs_clf = gs_clf.fit(norm_texts, semantic_tags)

    print()
    print("done in %0.3fs" % (time() - t0))
    print()
    print("Best score: %0.3f" % gs_clf.best_score_)
    print()
    print("Best parameters set:")

    best_estimator = gs_clf.best_estimator_
    best_parameters = best_estimator.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    f = open(model_output, 'wb')
    pickle.dump([best_estimator.steps[2][1],
                 best_estimator.steps[1][1],
                 best_estimator.steps[0][1]],
                f,
                -1)
    f.close()

    predicted = best_estimator.predict(norm_texts)

    best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))

    confidence_scores = best_estimator.decision_function(norm_texts)

    max_scores = []
    max_semantic_tags = []
    for scores in confidence_scores:
        max_score = -99999.99
        max_index = -1
        for i in range(0, len(scores)):
            score = scores[i]
            if max_score < score:
                max_score = score
                max_index = i

        max_scores.append(max_score)
        max_semantic_tags.append(best_estimator.steps[2][1].classes_[max_index])

    for i in range(0, len(norm_texts)):

        norm_text = norm_texts[i]
        semantic_tag = semantic_tags[i]
        predicted_semantic_tag = predicted[i]
        score = max_scores[i]

        lines.append(norm_text + '\t' +
                     semantic_tag + '\t' +
                     predicted_semantic_tag + '\t' +
                     str(score) + '\t' +
                     str(semantic_tag == predicted_semantic_tag) + '\n')

        if semantic_tag == predicted_semantic_tag:
            accurate_count += 1

    output = open(output, 'w')
    output.write('# Total texts: ' + str(len(lines)) + '\n')
    output.write('# Accurate texts: ' + str(accurate_count) + ' (' +  "{0:.2f}%".format(accurate_count / len(lines) * 100) + ')\n')
    output.write('\n')
    output.write('Normalized text\tSemantic tag\tPredicted semantic tag\tScore\tResult\n')
    for line in lines:
        output.write(line)
    output.close()
