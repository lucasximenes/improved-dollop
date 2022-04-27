from ExKMCbundle.ExKMC.Tree import Tree
from sklearn.datasets import make_blobs
import numpy as np
# # Create dataset
n = 100
d = 2
k = 2
X, _ = make_blobs(n_samples=n, n_features=d, centers=k)

# # Initialize tree with up to 6 leaves, predicting 3 clusters
tree = Tree(k=k, max_leaves=2*k) 

# # Construct the tree, and return cluster labels
prediction = tree.fit_predict(X)

# # Tree plot saved to filename
tree.plot('test')
