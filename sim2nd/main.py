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

def main():
    graph, labelsMat = loadgraph.GraphLoader().getGraph()
    name = str(random.randint(0,10000))

    transMat = transmat.TransMat(graph).deepWalkTransMat()
    walkSeries = randomwalk.RandomWalk(graph, transMat, walkNum = 10, walkLen = 80, name = name).nodeSeries(method = "deepwalk")
    model = embedding.GraphEmbedding(walkSeries, size = 128, window = 5, name = name).embedding(method = "deepwalk")
    resultMicro, resultMacro = train.Trainer(model, labelsMat, name = name).train(method = "deepwalk")
    print("deepwalk resultMicro : ", resultMicro)
    print("deepwalk resultMacro : ", resultMacro)

    transMat = transmat.TransMat(graph).sim2ndTransMat(lam = 1)
    names = name + '_1'
    walkSeries = randomwalk.RandomWalk(graph, transMat, walkNum = 10, walkLen = 80, name = names).nodeSeries(method = "sim2nd")
    model = embedding.GraphEmbedding(walkSeries, size = 128, window = 5, name = names).embedding(method = "sim2nd")
    resultMicro, resultMacro = train.Trainer(model, labelsMat, name = names).train(method = "sim2nd")
    print("sim2nd1 resultMicro : ", resultMicro)
    print("sim2nd1 resultMacro : ", resultMacro)

    transMat = transmat.TransMat(graph).sim2ndTransMat(lam = 0.5)
    names = name + '_0.5'
    walkSeries = randomwalk.RandomWalk(graph, transMat, walkNum = 10, walkLen = 80, name = names).nodeSeries(method = "sim2nd")
    model = embedding.GraphEmbedding(walkSeries, size = 128, window = 5, name = names).embedding(method = "sim2nd")
    resultMicro, resultMacro = train.Trainer(model, labelsMat, name = names).train(method = "sim2nd")
    print("sim2nd.5 resultMicro : ", resultMicro)
    print("sim2nd.5 resultMacro : ", resultMacro)

    transMat = transmat.TransMat(graph).sim2ndTransMat(lam = 0.1)
    names = name + '_0.1'
    walkSeries = randomwalk.RandomWalk(graph, transMat, walkNum = 10, walkLen = 80, name = names).nodeSeries(method = "sim2nd")
    model = embedding.GraphEmbedding(walkSeries, size = 128, window = 5, name = names).embedding(method = "sim2nd")
    resultMicro, resultMacro = train.Trainer(model, labelsMat, name = names).train(method = "sim2nd")
    print("sim2nd.1 resultMicro : ", resultMicro)
    print("sim2nd.1 resultMacro : ", resultMacro)

    transMat = transmat.TransMat(graph).node2vecTransMat(p = 0.25, q = 0.25)
    names = name + '_0.25_0.25'
    walkSeries = randomwalk.RandomWalk(graph, transMat, walkNum = 10, walkLen = 80, name = names).edgeSeries(method = "node2vec")
    model = embedding.GraphEmbedding(walkSeries, size = 128, window = 5, name = names).embedding(method = "node2vec")
    resultMicro, resultMacro = train.Trainer(model, labelsMat, name = names).train(method = "node2vec")
    print("node2vec resultMicro : ", resultMicro)
    print("node2vec resultMacro : ", resultMacro)
    return

if __name__ == '__main__':
    main()