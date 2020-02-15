# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 02:12:38 2020

@author: Hilbert
"""

import loadgraph
import transmat
import randomwalk
import embedding
import train
import numpy as np
import random
import os
import time

def Mkdir(method):
    if not os.path.exists('../data/{}'.format(method)):
        os.mkdir('../data/{}'.format(method))
        os.mkdir('../data/{}/transmat'.format(method))
        os.mkdir('../data/{}/walkseries'.format(method))
        os.mkdir('../data/{}/embvec'.format(method))
        os.mkdir('../data/{}/results'.format(method))
    return

def main():
    #load data
    graph, labelsMat = loadgraph.GraphLoader().getGraph()
    time.sleep(0.5)
    name = str(random.randint(0,10000)) # random name for mutiple runs of program.

    # deepwalk
    method = "deepwalk"
    print("Method = deepwalk")
    Mkdir(method)
    transMat = transmat.TransMat(graph).deepWalkTransMat()
    time.sleep(0.5)
    walkSeries = randomwalk.RandomWalk(graph, transMat, walkNum = 10, walkLen = 80, name = name).nodeSeries(method = method)
    model = embedding.GraphEmbedding(walkSeries, size = 128, window = 5, name = name).embedding(method = method)
    resultMicro, resultMacro = train.Trainer(model, labelsMat, name = name).train(method = method)
    print("deepwalk resultMicro : ", resultMicro)
    print("deepwalk resultMacro : ", resultMacro)

    # sim2nd
    method = "sim2nd"
    print("Method = sim2nd")
    Mkdir(method)
    transMat = transmat.TransMat(graph).sim2ndTransMat(lam = 1)
    time.sleep(0.5)
    names = name + '_1'
    walkSeries = randomwalk.RandomWalk(graph, transMat, walkNum = 10, walkLen = 80, name = names).nodeSeries(method = method)
    model = embedding.GraphEmbedding(walkSeries, size = 128, window = 5, name = names).embedding(method = method)
    resultMicro, resultMacro = train.Trainer(model, labelsMat, name = names).train(method = method)
    print("sim2nd resultMicro : ", resultMicro)
    print("sim2nd resultMacro : ", resultMacro)

    # node2vec
    method = "node2vec"
    print("Method = node2vec")
    Mkdir(method)
    transMat = transmat.TransMat(graph).node2vecTransMat(p = 0.25, q = 0.25)
    time.sleep(0.5)
    names = name + '_0.25_0.25'
    walkSeries = randomwalk.RandomWalk(graph, transMat, walkNum = 10, walkLen = 80, name = names).edgeSeries(method = method)
    model = embedding.GraphEmbedding(walkSeries, size = 128, window = 5, name = names).embedding(method = method)
    resultMicro, resultMacro = train.Trainer(model, labelsMat, name = names).train(method = method)
    print("node2vec resultMicro : ", resultMicro)
    print("node2vec resultMacro : ", resultMacro)
    return

if __name__ == '__main__':
    main()