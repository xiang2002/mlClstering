import numpy as np
from ..composition.rbf_knn import knn_torch as KNN
from ..composition.self_tuning import self_tuning
import torch
import torch
def lpp(data,
        n_dims=2,
        graph='knn',
        n_neighbors=10,
        sigma=None):
    '''
    :param data: (n_samples, n_features)
    :param n_dims: target dim
    :param n_neighbors: k nearest neighbors
    :param sigma: a param for rbf
    :return: data_ndim, eig_vec_picked
    '''
    N = data.shape[0]
    if graph=='knn':
        W = KNN(data, n_neighbors, sigma=sigma)
    elif graph =='self_tuning':
        W = self_tuning(data,k=n_neighbors)
    D = torch.zeros_like(W,device=data.device)

    for i in range(N):
        D[i, i] = torch.sum(W[i])

    L = D - W
    XDXT = data.T @ D @ data
    XLXT = data.T @ L @ data
    def inv_or_pinv(matrix):
        try:
        # 计算逆矩阵
            inv_matrix = torch.linalg.inv(matrix)
        # print("成功计算逆矩阵")
            return inv_matrix
        except RuntimeError as e:
        # 如果计算逆矩阵失败（例如矩阵是奇异矩阵），则计算伪逆
        # print("矩阵不可逆，计算伪逆")
            pinv_matrix = torch.linalg.pinv(matrix)
            return pinv_matrix
    Omega = torch.mm(inv_or_pinv(XDXT), XLXT)
    Omega += 1e-4 * torch.eye(Omega.shape[0], device=data.device)
    eig_val, eig_vec = torch.linalg.eigh(Omega)

    sort_index_ = torch.argsort(eig_val)
    eig_val = eig_val[sort_index_]
#    print(eig_val)
    j = 0
    sort_index_ = sort_index_[j:j+n_dims]

    return eig_vec[:, sort_index_]