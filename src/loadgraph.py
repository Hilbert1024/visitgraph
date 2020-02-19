# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:11:43 2020

@author: Hilbert1024
"""

from scipy.io import loadmat
import networkx as nx

class GraphLoader(object):
    """
    Load the graph. The type of loaded graph is based on the package networkx.

    Parameters
    ----------
    graphName : str
        The name of graph dataset.

    Notes
    -----
    The recognized graph names are:

    * ``blogcatalog``
    """
    def __init__(self, graphName = 'blogcatalog'):
        super(GraphLoader, self).__init__()
        self.graphName = graphName

    def getGraph(self):
        try:
            loadGraph = loadmat('../graph/{}.mat'.format(self.graphName))
            adjMat = loadGraph['network']
            graph = nx.from_numpy_matrix(adjMat.todense())
            labelsMat = loadGraph['group']
        except:
            raise("data not exists!")
        else:
            print('Graph {} is loaded.'.format(self.graphName))
            return graph, labelsMat
        