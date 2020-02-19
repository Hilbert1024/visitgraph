# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 22:29:50 2020

@author: Hilbert1024
"""

import numpy as np
import random

class RandomWalk(object):
    """
    Simulate a random walk series by given transition matrix.

    Parameters
    ----------
    graph : networkx.classes.graph.Graph
        A graph in networkx.
    transMat : dict
        The dictionary includes node:prob and edge:prob.
        The node:prob dictionary gets transition probablities from current node to its neighbors.
        The edge:prob dictionary gets transition probablities from previous node and current node to neighbors of current node.
    walkNum : int
        Numbers of random walks. Default is 10.
    walkLen : int
        Length of the series each time.
    name : str
        Name of file.
    """
    def __init__(self, graph, transMat, graphName, walkNum = 10, walkLen = 80, name = ""):
        super(RandomWalk, self).__init__()
        self.graph = graph
        self.nodes = graph.nodes()
        self.transMat = transMat
        self.walkNum = walkNum
        self.walkLen = walkLen
        self.graphName = graphName
        if name == "":
            self.name = str(random.randint(0,10000))
        else:
            self.name = name

    def _nodeChoice(self, probArr):
        """
        Generates a random sample from np.arange(len(probArr)).
        ProbArr is the probabilities associated with each entry in np.arange(len(probArr)).
        """
        probArr /= np.sum(probArr) #normalized
        return np.random.choice(len(probArr), p = probArr)

    def nodeSeries(self, method):
        """
        Simulate a random walk series when next movement only depends on current node, apply to deepwalk.
        """
        walks = []
        count = 0
        for _ in np.arange(self.walkNum):
            nodes = list(self.nodes)
            random.shuffle(nodes)
            for node in nodes:
                walk = [node]
                while len(walk) < self.walkLen:
                    curNode = walk[-1]
                    curNbr = list(self.graph.neighbors(curNode))
                    if len(curNbr) > 0:
                        walk.append(curNbr[self._nodeChoice(self.transMat[curNode])])
                    else:
                        break
                count += 1
                walks.append(walk)
                print('\r',"Simulating random walk series, process : {}%".format(round(100 * count / (self.walkNum * len(self.nodes)), 2)), end='', flush=True)
        try:
            np.save('../data/{}/{}/walkseries/walkseries_{}.npy'.format(self.graphName, method, self.name), walks)
        except FileNotFoundError:
            print("File can not found!")
        else:
            return walks

    def edgeSeries(self, method):
        """
        Simulate a random walk series when next movement only depends on current node, apply to node2vec.
        """

        walks = []
        count = 0
        for _ in np.arange(self.walkNum):
            nodes = list(self.nodes)
            random.shuffle(nodes)
            for node in nodes:
                walk = [node]
                while len(walk) < self.walkLen:
                    curNode = walk[-1]
                    curNbr = list(self.graph.neighbors(curNode))
                    if len(curNbr) > 0:
                        if len(walk) == 1: # First step walk to neighbors uniformly
                            nextNode = curNbr[self._nodeChoice([1 / len(curNbr)] * len(curNbr))]
                        else:
                            preNode = walk[-2]
                            nextNode = curNbr[self._nodeChoice(self.transMat[(preNode, curNode)])]
                        walk.append(nextNode)
                    else:
                        break
                count += 1
                walks.append(walk)
                print('\r',"Simulating random walk series, process : {}%".format(round(100 * count / (self.walkNum * len(self.nodes)), 2)), end='', flush=True)
        try:
            np.save('../data/{}/{}/walkseries/walkseries_{}.npy'.format(self.graphName, method, self.name), walks)
        except FileNotFoundError:
            print("File can not found!")
        else:
            return walks

    def nodeVisitSeries(self, method, alpha = 1):
        """
        Simulate a random walk series when next movement only depends on current node, apply to visitgraph.
        """
        walks = []
        count = 0
        visit = np.array([1] * len(self.nodes))
        for _ in np.arange(self.walkNum):
            nodes = list(self.nodes)
            random.shuffle(nodes)
            for node in nodes:
                walk = [node]
                while len(walk) < self.walkLen:
                    curNode = walk[-1]
                    curNbr = list(self.graph.neighbors(curNode))
                    if len(curNbr) > 0:
                        randomIndex = self._nodeChoice(self.transMat[curNode] * (1 / (visit[curNbr] ** alpha)))
                        nextNode = curNbr[randomIndex]
                        walk.append(nextNode)
                        visit[nextNode] += 1
                    else:
                        break
                count += 1
                walks.append(walk)
                print('\r',"Simulating random walk series, process : {}%".format(round(100 * count / (self.walkNum * len(self.nodes)), 2)), end='', flush=True)
        try:
            np.save('../data/{}/{}/walkseries/walkseries_{}.npy'.format(self.graphName, method, self.name), walks)
        except FileNotFoundError:
            print("File can not found!")
        else:
            return walks