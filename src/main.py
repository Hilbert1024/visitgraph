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

def Mkdir(graphName, method):
    if not os.path.exists('../data/{}'.format(graphName)):
        os.mkdir('../data/{}'.format(graphName))
    if not os.path.exists('../data/{}/{}'.format(graphName, method)):
        os.mkdir('../data/{}/{}'.format(graphName, method))
        os.mkdir('../data/{}/{}/transmat'.format(graphName, method))
        os.mkdir('../data/{}/{}/walkseries'.format(graphName, method))
        os.mkdir('../data/{}/{}/embvec'.format(graphName, method))
        os.mkdir('../data/{}/{}/results'.format(graphName, method))
    return

def PlotF1(graphName, code):
    if not os.path.exists('../figure/{}'.format(graphName)):
        os.mkdir('../figure/{}'.format(graphName))
    try:
        dwMi = np.load('../data/{}/deepwalk/results/resultsMicro_{}.npy'.format(graphName, code))
        n2vMi = np.load('../data/{}/node2vec/results/resultsMicro_{}_0.25_0.25.npy'.format(graphName, code))
        vgMi = np.load('../data/{}/visitgraph/results/resultsMicro_{}.npy'.format(graphName, code))

        dwMa = np.load('../data/{}/deepwalk/results/resultsMacro_{}.npy'.format(graphName, code))
        n2vMa = np.load('../data/{}/node2vec/results/resultsMacro_{}_0.25_0.25.npy'.format(graphName, code))
        vgMa = np.load('../data/{}/visitgraph/results/resultsMacro_{}.npy'.format(graphName, code))

        dwMicro = np.mean(np.split(dwMi[0],9),axis = 1)
        n2vMicro = np.mean(np.split(n2vMi[0],9),axis = 1)
        vgMicro = np.mean(np.split(vgMi[0],9),axis = 1)

        dwMacro = np.mean(np.split(dwMa[0],9),axis = 1)
        n2vMacro = np.mean(np.split(n2vMa[0],9),axis = 1)
        vgMacro = np.mean(np.split(vgMa[0],9),axis = 1)

        plt.style.use('ggplot') #设置ggplot绘图风格
        fig = plt.figure(figsize = (15, 5))
        xlab = np.arange(0.1,1,0.1)
        ax1 = fig.add_subplot(121)
        l1, = ax1.plot(xlab, dwMicro, 's-')
        l2, = ax1.plot(xlab, n2vMicro, 'd-')
        l3, = ax1.plot(xlab, vgMicro, '^-')
        ax1.legend([l1,l2,l3],['deepwalk','node2vec','visitgraph'], loc = 0)
        ax1.set_xlabel("Train ratio")
        ax1.set_ylabel("Micro-F1")

        ax2 = fig.add_subplot(122)
        l1, = ax2.plot(xlab, dwMacro, 's-')
        l2, = ax2.plot(xlab, n2vMacro, 'd-')
        l3, = ax2.plot(xlab, vgMacro, '^-')
        ax2.legend([l1,l2,l3],['deepwalk','node2vec','visitgraph'], loc = 0)
        ax2.set_xlabel("Train ratio")
        ax2.set_ylabel("Maicro-F1")
        fig.suptitle(graphName, fontsize=16)
    except:
        pass
    else:
        plt.savefig("../figure/{}/test_{}.jpg".format(graphName, code), dpi = 300)
    return

def main():
    #load data
    graphNameList = ['Homo_sapiens','wikipedia','blogcatalog']
    node2vecPara = {'Homo_sapiens' : (4, 1), 'wikipedia' : (4, 0.5), 'blogcatalog' : (0.25, 0.25)}
    visitgraphPara = {'Homo_sapiens' : 0.5, 'wikipedia' : 0.1, 'blogcatalog' : 1}
    for graphName in graphNameList:
        graph, labelsMat = loadgraph.GraphLoader(graphName = graphName).getGraph()
        time.sleep(0.5)
        name = str(random.randint(0,10000)) # random name for mutiple runs of program.
        window = 10
        print("Data set : {}".format(graphName))
        print("Experiments code : {}.".format(name))

        # visitgraph
        alpha = visitgraphPara[graphName]
        tempTime = time.time()
        method = "visitgraph"
        print("Method = visitgraph")
        Mkdir(graphName, method)
        transMat = transmat.TransMat(graph, graphName).unnormTransMat(method = method)
        time.sleep(0.5)
        walkSeries = randomwalk.RandomWalk(graph, transMat, walkNum = 10, walkLen = 80, graphName = graphName, name = name).nodeVisitSeries(method = method, alpha = alpha)
        model = embedding.GraphEmbedding(walkSeries, size = 128, window = window, graphName = graphName, name = name).embedding(method = method)
        resultMicro, resultMacro = train.Trainer(model, labelsMat, graphName = graphName, name = name).train(method = method)
        print("\n{} resultMicro : ".format(method), np.mean(np.split(resultMicro[0],9), axis = 1))
        print("{} resultMacro : ".format(method), np.mean(np.split(resultMacro[0],9), axis = 1))
        print("{} costs {}s.\n".format(method, round(time.time() - tempTime)))

        # deepwalk
        tempTime = time.time()
        method = "deepwalk"
        print("Method = deepwalk")
        Mkdir(graphName, method)
        transMat = transmat.TransMat(graph, graphName).unnormTransMat(method = method)
        time.sleep(0.5)
        walkSeries = randomwalk.RandomWalk(graph, transMat, walkNum = 10, walkLen = 80, graphName = graphName, name = name).nodeSeries(method = method)
        model = embedding.GraphEmbedding(walkSeries, size = 128, window = window, graphName = graphName, name = name).embedding(method = method)
        resultMicro, resultMacro = train.Trainer(model, labelsMat, graphName = graphName, name = name).train(method = method)
        print("\n{} resultMicro : ".format(method), np.mean(np.split(resultMicro[0],9), axis = 1))
        print("{} resultMacro : ".format(method), np.mean(np.split(resultMacro[0],9), axis = 1))
        print("{} costs {}s.\n".format(method, round(time.time() - tempTime)))

        # node2vec
        p, q = node2vecPara[graphName]
        tempTime = time.time()
        method = "node2vec"
        print("Method = node2vec")
        Mkdir(graphName, method)
        transMat = transmat.TransMat(graph, graphName).node2vecTransMat(p, q)
        time.sleep(0.5)
        names = name + '_0.25_0.25'
        walkSeries = randomwalk.RandomWalk(graph, transMat, walkNum = 10, walkLen = 80, graphName = graphName, name = names).edgeSeries(method = method)
        model = embedding.GraphEmbedding(walkSeries, size = 128, window = window, graphName = graphName, name = names).embedding(method = method)
        resultMicro, resultMacro = train.Trainer(model, labelsMat, graphName = graphName, name = names).train(method = method)
        print("\n{} resultMicro : ".format(method), np.mean(np.split(resultMicro[0], 9), axis = 1))
        print("{} resultMacro : ".format(method), np.mean(np.split(resultMacro[0], 9), axis = 1))
        print("{} costs {}s.\n".format(method, round(time.time() - tempTime)))

        # plot
        PlotF1(graphName, name)
    return

if __name__ == '__main__':
    main()