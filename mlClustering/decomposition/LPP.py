import numpy as np
from ..composition.rbf_knn import knn_torch as KNN
import torch
#import sys
#print(sys.path)
def lpp(data,
        n_dims=2,
        n_neighbors=10, sigma=1.0):
    '''
    :param data: (n_samples, n_features)
    :param n_dims: target dim
    :param n_neighbors: k nearest neighbors
    :param sigma: a param for rbf
    :return: data_ndim, eig_vec_picked
    '''
    N = data.shape[0]
    W = KNN(data, n_neighbors, sigma=None)
    D = torch.zeros_like(W,device=data.device)

    for i in range(N):
        D[i, i] = torch.sum(W[i])

    L = D - W
    XDXT = torch.mm(torch.mm(data.t(), D), data)
    XLXT = torch.mm(torch.mm(data.t(), L), data)

    eig_val, eig_vec = torch.linalg.eigh(torch.mm(torch.linalg.pinv(XDXT), XLXT))

    sort_index_ = torch.argsort(eig_val)
    eig_val = eig_val[sort_index_]

    j = 0
    sort_index_ = sort_index_[j:j+n_dims]

    return eig_vec[:, sort_index_]