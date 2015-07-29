# coding: utf-8
# Copyright (C) 2015 Marinka Zitnik <marinka.zitnik@fri.uni-lj.si>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
COPACAR Python package

This package provides routines to compute COPACAR collective pairwise classification
model.

COPACAR factors a collection of relation data matrices (a tensor) $X^{(k)},$ $k = 1,2,\dots,m$, such that each
relation $X^{(k)}$ is factored into

.. math:: X^{(k)} = A * R^{(k)} * A.T,

where $A$ is a latent matrix shared across relations and $R^{(k)}$ is a latent component
interaction matrix. The latent model is found as to minimize

.. math:: \sum_{i,j,g,h} (X^{(k)}_ij - X^{(k)}_gh) \log \sigma(A_i^T * R^{(k)} * A_j - A_g^T * R^{(k)} * A_h).

The relations are $N \times N$ matrices. Usually, these
matrices correspond to the adjacency matrices of the relational graph
for a particular relation in a multi-relational data set.

See
---
For a full description of the algorithm see:
.. [1] Marinka Zitnik, Blaz Zupan,
       "Collective pairwise classification for multi-way analysis of disease and drug data"
"""

from .copacar import copacar as copacar_sgd
