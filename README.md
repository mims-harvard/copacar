COPACAR
=======
 
COPACAR is a generic collective pairwise classification model for multi-way data analysis. COPACAR combines the superiority of factorization models in large relational data domains with the classification capabilities based on using the pairwise ranking loss. In contrast to current collective relational approaches COPACAR aims to infer probabilities so that those for existing relationships are higher than for assumed-to-be-negative relationships. 

Although COPACAR bears correspondence with the maximization of non-differentiable area under the ROC curve, it comes with a learning algorithm that scales well on multi-relational data. 

COPACAR can make *category-jumping* predictions about, for example, diseases from genomic and clinical data generated far outside the molecular context.   

This repository contains supplementary material for *Collective pairwise classification for multi-way analysis of disease and drug data* by Marinka Zitnik and Blaz Zupan.
 
 
About COPACAR
-------------
COPACAR factors a collection of relation data matrices (a tensor) `X^(k),` `k = 1,2,...,m`, such that each
relation `X^(k)` is factored into

    X^(k) = A * R^(k) * A.T,

where `A` is the matrix storing latent components of entities in the domain. `A` is shared across relations. 
Matrix `R^(k)` is the latent component interaction matrix. COPACAR optimizes for the latent model 
w.r.t. ranking based on pairwise classification

    sum_{i,j,g,h} [ (X^(k)_ij - X^(k)_gh) * log(sigma(A_i^T * R^(k) * A_j - A_g^T * R^(k) * A_h)) + reg ].

The relations are `N x N` matrices. Usually, these
matrices correspond to the adjacency matrices of the relational graph
for a particular relation in a multi-relational data set.


Dependencies
------------
The required dependencies to build the software are `Numpy >= 1.8`, `SciPy >= 0.10`, `Joblib >= 0.8`, `scikit-learn >= 0.12`.


Usage
-----
Example script to classify kinships data using COPACAR:

```python
import logging
from scipy.io.matlab import loadmat
from scipy.sparse import lil_matrix
from copacar import copacar_sgd

# Set logging to INFO to see COPACAR information
logging.basicConfig(level=logging.INFO)

# Load Matlab data and convert it to dense tensor format
T = loadmat('data/alyawarradata.mat')['Rs']
X = [lil_matrix(T[:, :, k]) for k in range(T.shape[2])]

# Decompose multi-way data using COPACAR-SGD
A, R, fit, itr, exectimes = copacar_sgd(X, 10, lambda_A=1e-1, lambda_R=1e-1, max_iter=100, n_jobs=-1)
```

For more examples on the usage of COPACAR, please see the [examples](examples) directory. 


Install
-------
To install in your home directory, use

    python setup.py install --user

To install for all users on Unix/Linux

    python setup.py build
    sudo python setup.py install

To install in development mode

    python setup.py develop


Citing
------

    @inproceedings{Zitnik2016,
      title     = {Collective pairwise classification for multi-way analysis of disease and drug data},
      author    = {{\v{Z}}itnik, Marinka and Zupan, Bla{\v{z}}},
      booktitle = {Pacific Symposium on Biocomputing},
      volume    = {21},
      pages     = {},
      year      = {2016}
    }
    
    
License
-------
COPACAR is licensed under the GPLv2.
