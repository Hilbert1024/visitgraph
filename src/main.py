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
import random
import os
import time
import numpy as np
import matplotlib.pyplot as plt

def Mkdir(method):
    if not os.path.exists('../data/{}'.format(method)):
        os.mkdir('../data/{}'.format(method))
        os.mkdir('../data/{}/transmat'.format(method))
        os.mkdir('../data/{}/walkseries'.format(method))
        os.mkdir('../data/{}/embvec'.format(method))
        os.mkdir('../data/{}/results'.format(method))
    return

def PlotF1(code):
    try:
        dwMi = np.load('../data/deepwalk/results/resultsMicro_{}.npy'.format(code))
        n2vMi = np.load('../data/node2vec/results/resultsMicro_{}_0.25_0.25.npy'.format(code))
        vgMi = np.load('../data/visitgraph/results/resultsMicro_{}.npy'.format(code))

        dwMa = np.load('../data/deepwalk/results/resultsMacro_{}.npy'.format(code))
        n2vMa = np.load('../data/node2vec/results/resultsMacro_{}_0.25_0.25.npy'.format(code))
        vgMa = np.load('../data/visitgraph/results/resultsMacro_{}.npy'.format(code))

        dwMicro = np.mean(np.split(dwMi[0],9),axis = 1)
        n2vMicro = np.mean(np.split(n2vMi[0],9),axis = 1)
        sim2ndMicro = np.mean(np.split(vgMi[0],9),axis = 1)

        dwMacro = np.mean(np.split(dwMa[0],9),axis = 1)
        n2vMacro = np.mean(np.split(n2vMa[0],9),axis = 1)
        sim2ndMacro = np.mean(np.split(vgMa[0],9),axis = 1)

        plt.style.use('ggplot') #设置ggplot绘图风格
        fig = plt.figure(figsize = (15, 5))
        xlab = np.arange(0.1,1,0.1)
        ax1 = fig.add_subplot(121)
        l1, = ax1.plot(xlab, dwMicro, '^-')
        l2, = ax1.plot(xlab, n2vMicro, '^-')
        l3, = ax1.plot(xlab, sim2ndMicro, '^-')
        ax1.legend([l1,l2,l3],['deepwalk','node2vec','visitgraph'], loc = 0)
        ax1.set_title("Micro-F1")

        ax2 = fig.add_subplot(122)
        l1, = ax2.plot(xlab, dwMacro, '^-')
        l2, = ax2.plot(xlab, n2vMacro, '^-')
        l3, = ax2.plot(xlab, sim2ndMacro, '^-')
        ax2.legend([l1,l2,l3],['deepwalk','node2vec','visitgraph'], loc = 0)
        ax2.set_title("Macro-F1")
    except:
        pass
    else:
        plt.savefig("../figure/test_{}.jpg".format(code), dpi = 300)
    return

def main():
    #load data
    graph, labelsMat = loadgraph.GraphLoader().getGraph()
    time.sleep(0.5)
    name = str(random.randint(0,10000)) # random name for mutiple runs of program.
    window = 10
    print("Experiments code : {}.".format(name))

    # deepwalk
    tempTime = time.time()
    method = "deepwalk"
    print("Method = deepwalk")
    Mkdir(method)
    transMat = transmat.TransMat(graph).deepWalkTransMat()
    time.sleep(0.5)
    walkSeries = randomwalk.RandomWalk(graph, transMat, walkNum = 10, walkLen = 80, name = name).nodeSeries(method = method)
    model = embedding.GraphEmbedding(walkSeries, size = 128, window = window, name = name).embedding(method = method)
    resultMicro, resultMacro = train.Trainer(model, labelsMat, name = name).train(method = method)
    print("\n{} resultMicro : ".format(method), np.mean(np.split(resultMicro[0],9), axis = 1))
    print("{} resultMacro : ".format(method), np.mean(np.split(resultMacro[0],9), axis = 1))
    print("{} costs {}s.\n".format(method, round(time.time() - tempTime)))

    # node2vec
    tempTime = time.time()
    method = "node2vec"
    print("Method = node2vec")
    Mkdir(method)
    transMat = transmat.TransMat(graph).node2vecTransMat(p = 0.25, q = 0.25)
    time.sleep(0.5)
    names = name + '_0.25_0.25'
    walkSeries = randomwalk.RandomWalk(graph, transMat, walkNum = 10, walkLen = 80, name = names).edgeSeries(method = method)
    model = embedding.GraphEmbedding(walkSeries, size = 128, window = window, name = names).embedding(method = method)
    resultMicro, resultMacro = train.Trainer(model, labelsMat, name = names).train(method = method)
    print("\n{} resultMicro : ".format(method), np.mean(np.split(resultMicro[0], 9), axis = 1))
    print("{} resultMacro : ".format(method), np.mean(np.split(resultMacro[0], 9), axis = 1))
    print("{} costs {}s.\n".format(method, round(time.time() - tempTime)))
    
    # visitgraph
    tempTime = time.time()
    method = "visitgraph"
    print("Method = visitgraph")
    Mkdir(method)
    transMat = transmat.TransMat(graph).visitgraphTransMat()
    time.sleep(0.5)
    walkSeries = randomwalk.RandomWalk(graph, transMat, walkNum = 10, walkLen = 80, name = name).nodeVisitSeries(method = method)
    model = embedding.GraphEmbedding(walkSeries, size = 128, window = window, name = name).embedding(method = method)
    resultMicro, resultMacro = train.Trainer(model, labelsMat, name = name).train(method = method)
    print("\n{} resultMicro : ".format(method), np.mean(np.split(resultMicro[0],9), axis = 1))
    print("{} resultMacro : ".format(method), np.mean(np.split(resultMacro[0],9), axis = 1))
    print("{} costs {}s.\n".format(method, round(time.time() - tempTime)))

    # plot
    PlotF1(name)
    return

if __name__ == '__main__':
    main()