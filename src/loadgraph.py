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
        if self.graphName == 'blogcatalog':
            loadGraph = loadmat('../graph/blogcatalog.mat')
            adjMat = loadGraph['network']
            graph = nx.from_numpy_matrix(adjMat.todense())
            labelsMat = loadGraph['group']
            print('Graph blogcatalog is loaded.')
            return graph, labelsMat
        else:
            raise("data not exists!")
        