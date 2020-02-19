# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 01:50:00 2020

@author: Hilbert1024
"""

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.utils import shuffle as skshuffle
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

class TopKRanker(OneVsRestClassifier):

    def predict(self, X, topKList):
        assert X.shape[0] == len(topKList)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        probLabel = np.zeros((probs.shape[0],probs.shape[1]))
        for i, k in enumerate(topKList):
            probs_ = probs[i, :]
            probLabel[i, probs_.argsort()[-k:]] = 1
        return probLabel

class Trainer(object):
    """
    Apply OneVsRestClassifier of logistic regression to multi-labels classification.
    """
    def __init__(self, model, labelsMat, graphName, name = ""):
        super(Trainer, self).__init__()
        self.model = model
        self.labelsMat = labelsMat
        self.featureMat = np.asarray([model[str(node)] for node in np.arange(labelsMat.shape[0])])
        self.graphName = graphName 
        if name == "":
            self.name = str(random.randint(0,10000))
        else:
            self.name = name

    def train(self, method):
        trainRatio = np.arange(0.1,1,0.1)
        resultMicro, resultMacro = [], []
        for ratio in trainRatio:
            print('\r',"Training with {}% labeled nodes...".format(int(100 * ratio)), end='', flush=True)
            resultMicroMat, resultMacroMat = [], []
            for _ in np.arange(10): # Train 10 times.
                X, y = skshuffle(self.featureMat, self.labelsMat)
                trainSize = int(ratio * X.shape[0])
                X_train = X[:trainSize]
                y_train = y[:trainSize]
                X_test = X[trainSize:]
                y_test = y[trainSize:] 
                clf = TopKRanker(LogisticRegression(solver = 'liblinear'))
                clf.fit(X_train, y_train)
                topKList = np.diff(y_test.tocsr().indptr)
                preds = clf.predict(X_test, topKList)
                resultMicro.append(f1_score(y_test, preds, average = 'micro'))
                resultMacro.append(f1_score(y_test, preds, average = 'macro'))
            resultMicroMat.append(resultMicro)
            resultMacroMat.append(resultMacro)
            
        np.save('../data/{}/{}/results/resultsMicro_{}.npy'.format(self.graphName, method, self.name), np.array(resultMicroMat))
        np.save('../data/{}/{}/results/resultsMacro_{}.npy'.format(self.graphName, method, self.name), np.array(resultMacroMat))
        return np.array(resultMicroMat), np.array(resultMacroMat)