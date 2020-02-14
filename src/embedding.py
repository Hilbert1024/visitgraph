# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 01:25:52 2020

@author: Hilbert1024
"""
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

class GraphEmbedding(object):
    """
    Parameters
    ----------
    walkSeries : iterable of iterables
        The random walk series.
    size : int
        Dimensionality of the word vectors. Default is 128.
    window : int, optional
        Maximum distance between the current and predicted word within a sentence. Default is 5.
    name : str
        Name of file.
    Reference
    ---------
    Goldberg Y, Levy O. word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method[J]. 
    arXiv preprint arXiv:1402.3722, 2014.
    """
    def __init__(self, walkSeries, size = 128, window = 5, name = ""):
        super(GraphEmbedding, self).__init__()
        self.walkSeries = walkSeries
        self.size = size
        self.window = window
        if name == "":
            self.name = str(random.randint(0,10000))
        else:
            self.name = name

    def _int2str(self):
        f = lambda x:[str(y) for y in x]
        return list(map(f, self.walkSeries))

    def embedding(self, method):
        walks = self._int2str()
        model = Word2Vec(walks, size = self.size, window = self.window, min_count=0) # Can not ignore any nodes.
        model.wv.save_word2vec_format('../data/{}/embvec/embvec_{}.txt'.format(method, self.name))
        model = KeyedVectors.load_word2vec_format('../data/{}/embvec/embvec_{}.txt'.format(method, self.name), binary=False)
        return model

