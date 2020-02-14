# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:26:08 2020

@author: Hilbert1024
"""
import numpy as np
import os
import pickle

class TransMat(object):
    """
    Create the transition matrix of random walk on graph.
    
    Parameters
    ----------
    graph : networkx.classes.graph.Graph
        A graph in networkx.

    Notes
    -----
    The graph embedding methods are:

    * ``deepwalk``
    * ``node2vec``
    * ``sim2nd``
    """
    def __init__(self, graph):
        super(TransMat, self).__init__()
        self.graph = graph
        self.nodes = graph.nodes()

    def _getEdgeProb(self, preNode, curNode, p, q):
        unnormProb = []
        for curNbr in self.graph.neighbors(curNode):
            if curNbr == preNode:
                unnormProb.append(self.graph[curNode][curNbr]['weight'] / p)
            elif self.graph.has_edge(curNbr, preNode):
                unnormProb.append(self.graph[curNode][curNbr]['weight'])
            else:
                unnormProb.append(self.graph[curNode][curNbr]['weight'] / q)
        return np.array(unnormProb) / np.sum(unnormProb)

    def _secondSimilarity(self, curNode, nextNode):
        curNeighbor = self.graph.neighbors(curNode)
        curNeighborLen = curNeighbor.__length_hint__()
        nextNeighbor = self.graph.neighbors(nextNode)
        nextNeighborLen = nextNeighbor.__length_hint__()
        count = 0
        if curNeighborLen < nextNeighborLen:
            for nodes in curNeighbor:
                if nodes in nextNeighbor:
                    count += 1
        else:
            for nodes in nextNeighbor:
                if nodes in curNeighbor:
                    count += 1
        return count/np.sqrt(curNeighborLen*nextNeighborLen)

    def _get2ndSimMat(self):
        if not os.path.exists('../data/sim2nd/transmat/secondsim.pkl'):
            edgeSecondSim = {}
            for node in self.nodes:
                nodeNeighbor = self.graph.neighbors(node)
                for neighbor in nodeNeighbor:
                    if neighbor > node and (node,neighbor) not in edgeSecondSim.keys():
                        edgeSecondSim[(node, neighbor)] = self._secondSimilarity(node,neighbor)
                    elif neighbor < node and (neighbor, node) not in edgeSecondSim.keys():
                        edgeSecondSim[(neighbor, node)] = self._secondSimilarity(node,neighbor)
            with open('../data/sim2nd/transmat/secondsim.pkl', 'wb') as outp:
                pickle.dump(edgeSecondSim, outp)
        else:
            with open('../data/sim2nd/transmat/secondsim.pkl', 'rb') as inp:
                edgeSecondSim = pickle.load(inp)
        return edgeSecondSim

    def _secondSimProb(self, curNode, lam, edgeSecondSim):
        unnormNeighborSim = []
        for neighbor in self.graph.neighbors(curNode):
            try:
                tempSim = edgeSecondSim[(curNode,neighbor)]
            except KeyError:
                unnormNeighborSim.append(edgeSecondSim[(neighbor,curNode)])
            else:
                unnormNeighborSim.append(tempSim)
        z = np.max(unnormNeighborSim) - np.min(unnormNeighborSim)
        if z == 0:
            return np.array([1/len(unnormNeighborSim)]*len(unnormNeighborSim))
        unnormNeighborSim = 1 / (unnormNeighborSim + lam*z)
        return unnormNeighborSim / np.sum(unnormNeighborSim)

    def deepWalkTransMat(self):
        """
        Reference
        ---------
        Perozzi B, Al-Rfou R, Skiena S. Deepwalk: Online learning of social representations[C]
        Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. 2014: 701-710.
        """
        if not os.path.exists('../data/deepwalk/transmat/transmat.pkl'):
            nodeTrans = dict()
            for node in self.nodes:
                unnormProb = np.array([self.graph[node][nbr]['weight'] for nbr in self.graph.neighbors(node)])
                nodeTrans[node] = unnormProb / np.sum(unnormProb)
            with open('../data/deepwalk/transmat/transmat.pkl', 'wb') as outp:
                pickle.dump(nodeTrans, outp)
        else:
            with open('../data/deepwalk/transmat/transmat.pkl', 'rb') as inp:
                nodeTrans = pickle.load(inp)
        return nodeTrans

    def node2vecTransMat(self, p, q, directed = False):
        """
        Parameters
        ----------
        p,q : float,float
            Defined in the paper.
        directed : boolean
            True if the graph is directed. Default is False.

        Reference
        ---------
        Grover A, Leskovec J. node2vec: Scalable feature learning for networks[C]
        Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. 2016: 855-864.

        Notes
        -----
        This process may cost much time.
        """
        if not os.path.exists('../data/node2vec/transmat/transmat_{}_{}.pkl'.format(p,q)):
            edgeTrans = dict()
            if directed:
                for edge in self.graph.edges():
                    edgeTrans[edge] = self._getEdgeProb(edge[0], edge[1], p, q)
            else:
                for edge in self.graph.edges():
                    edgeTrans[edge] = self._getEdgeProb(edge[0], edge[1], p, q)
                    edgeTrans[(edge[1], edge[0])] = self._getEdgeProb(edge[1], edge[0], p, q)
            with open('../data/node2vec/transmat/transmat_{}_{}.pkl'.format(p,q), 'wb') as outp:
                pickle.dump(edgeTrans, outp)
        else:
            with open('../data/node2vec/transmat/transmat_{}_{}.pkl'.format(p,q), 'rb') as inp:
                edgeTrans = pickle.load(inp)
        return edgeTrans

    def sim2ndTransMat(self, lam):
        if not os.path.exists('../data/sim2nd/transmat/transmat.pkl'):
            nodeTrans = {}
            edgeSecondSim = self._get2ndSimMat()
            for node in self.nodes:
                nodeTrans[node] = self._secondSimProb(node, lam, edgeSecondSim)
            with open('../data/sim2nd/transmat/transmat_{}.pkl'.format(lam), 'wb') as outp:
                pickle.dump(nodeTrans, outp)
        else:
            with open('../data/node2vec/transmat/transmat_{}.pkl'.format(lam), 'rb') as inp:
                nodeTrans = pickle.load(inp)
        return nodeTrans
