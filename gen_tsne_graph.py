#!/usr/bin/env python3
import os
import sys
import pickle
import random

from time import time

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import offsetbox

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.manifold import TSNE

def plot_embedding(X, y, title = None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0],
                 X[i, 1],
                 str(y[i]),
                 color = plt.cm.Set1(y[i] / 2.),
                 fontdict = {'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

def main():

    training_data_path = sys.argv[1]

    texts = []
    labels = []

    training_data = open(training_data_path, 'r')
    for line in training_data:
        arr = line.strip().split('\t')
        text = arr[0]
        label = arr[1]

        texts.append(text)
        labels.append(label)

    y = []
    for label in labels:
        if label == 'promotion ที่ใช้อยู่':
            y.append(1)
        else:
            y.append(0)

    count_vect = CountVectorizer(tokenizer = str.split, ngram_range = (1, 1))
    tfidf_transformer = TfidfTransformer()

    X_train = count_vect.fit_transform(texts)
    X_train = tfidf_transformer.fit_transform(X_train)

    tsne = TSNE()
    t0 = time()
    X_tsne = tsne.fit_transform(X_train.toarray())

    plot_embedding(X_tsne,
                   y,
                   "t-SNE embedding of the texts (time %.2fs)" % (time() - t0))

    plt.show()

if __name__ == '__main__':
    main()
