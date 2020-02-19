# visitgraph
## Introduction
visitgraph is a random walk strategy on graph. It marks the visited nodes when walking and tends to walk on the nodes that with less visits.

In current node, deepwalk[1] walks to its neighbors with equal probabilities, while node2vec[2] prefer to walk around current node(Breadth First Search) and far away from current node(Depth First Search). However, different from the above two methods, visitgraph keep a list which records the number of visits of each node. When walking to next node, visitgraph gets the number of visits of its neighbors in list, then walks to its neighbors with probabilities that are positively correlated with the reciprocal of the visiting times. Visitgraph performs better than deepwalk and node2vec on blogcatalog. The results are as follows.

You can also consider this program as a graph embedding framework which contains simulating random walk series, embedding, training. You may add other methods of random walk in ./src/randomwalk.py.

Version 1.0.0 2020-2-18.

Version 2.0.0 2020-2-19.
1. Add a parameter density index in visitgraph. 
2. Add 2 datasets, PPI(Homo Sapiens) and wikipedia.
3. Delete results.

## Requirements

+ numpy
+ networkx
+ gensim
+ scipy
+ scikit_learn
+ matplotlib

## Start
```markdown
cd visitgraph
pip install -r requirements.txt
cd src
python main.py
```

## Parameter settings

The parameter settings used for visitgraph are in line with typical values used for deepwalk and node2vec. Specifically, we set d = 128,
r = 10, l = 80, w = 10 where
+ d : embedding size.
+ r : walks per nodes.
+ l : walk length.
+ w : window size.

## Dataset

|  Data set   | # nodes  | # edges | # lables | type | density |
|  ----  | ----  | ----| ---- | ---- | ---- |
| blogcatalog  | 10312 | 333983 | 39 | undirected, unweighted | $6.28 \times 10^{-3}$ |
| PPI(Homo Sapiens)  | 3890 | 76584 | 50 | undirected, unweighted | $1.01 \times 10^{-2}$ |
| wikipedia | 4777 | 184812 | 40 | undirected, weighted | $1.62 \times 10^{-2}$ |

## Notes
You may cost much time on generating node2vec transition matrix.

## References
1. Perozzi B, Al-Rfou R, Skiena S. Deepwalk: Online learning of social representations[C]//Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. 2014: 701-710.
2. Grover A, Leskovec J. node2vec: Scalable feature learning for networks[C]//Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. 2016: 855-864.