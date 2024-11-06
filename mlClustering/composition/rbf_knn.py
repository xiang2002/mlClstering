import numpy as np
import torch

def knn_numpy(data, n_neighbors=10, sigma=None):
    """
    Computes the similarity matrix W using K-nearest neighbors and the Gaussian kernel function.

    Parameters:
    data (np.ndarray): Input data matrix.
    n_neighbors (int): Number of neighbors to consider for each point.
    sigma (float, optional): Standard deviation for the Gaussian kernel. If None, it is computed from the data.

    Returns:
    np.ndarray: The similarity matrix W.
    """
    def rbf(dist, sigma=None):
        """
        rbf kernel function
        """
        if sigma is None:
            sigma = np.sqrt(dist).mean()
        return np.exp(-(dist / 2 * sigma**2))

    def cal_pairwise_dist(x):
        """
        计算pairwise 距离, x是matrix
        (a-b)^2 = a^2 + b^2 - 2*a*b
        """
        sum_x = np.sum(np.square(x), 1)
        dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
        # 返回任意两个点之间距离的平方
        return dist

    dist = cal_pairwise_dist(data)
    dist[dist < 0] = 0
    n = dist.shape[0]
    rbf_dist = rbf(dist, sigma)

    W = np.zeros((n, n))
    for i in range(n):
        index_ = np.argsort(dist[i])[1 : 1 + n_neighbors]
        W[i, index_] = rbf_dist[i, index_]
        W[index_, i] = rbf_dist[index_, i]

    return W


def knn_torch(data, n_neighbors=10, sigma=None):
    """
    Computes the similarity matrix S using K-nearest neighbors and the Gaussian kernel function.
    
    Parameters:
    data (torch.Tensor): Input data matrix.
    n_neighbors (int): Number of neighbors to consider for each point.
    sigma (float, optional): Standard deviation for the Gaussian kernel. If None, it is computed from the data.

    Returns:
    torch.Tensor: The similarity matrix S.
    """

    def rbf(dist, sigma=2):
        """
        rbf kernel function
        """
        if sigma is None:
            sigma = torch.sqrt(dist).mean()
        # print("sigma",sigma)
        return torch.exp(-(dist / 2 * sigma**2))

    def cal_pairwise_dist(X):
        """
        计算pairwise 距离, x是matrix
        (a-b)^2 = a^2 + b^2 - 2*a*b
        """
        # 计算 X 中每个样本之间的欧氏距离平方
        X_square = torch.sum(X**2, dim=1).reshape(-1, 1)  # 每个样本的平方和
        distances_square = X_square - 2 * X @ X.T + X_square.T  # 欧氏距离的平方矩阵

        # 确保数值稳定，防止微小负数
        distances_square = torch.clamp(distances_square, min=0.0)

        # print("欧氏距离的平方的完整矩阵：")
        # print(distances_square)
        return distances_square

    dist = cal_pairwise_dist(data)
    # print(dist.max(),dist.min())
    dist[dist < 0] = 0
    n = dist.shape[0]
    rbf_dist = rbf(dist, sigma)

    S = torch.zeros_like(dist,device=data.device)
    for i in range(n):
        index_ = torch.argsort(dist[i])[
            1 : 1 + n_neighbors
        ]  # 最小的距离是0（自己跟自己）所以从1开始索引
        S[i, index_] = rbf_dist[i, index_]
        S[index_, i] = rbf_dist[index_, i]

    return S
