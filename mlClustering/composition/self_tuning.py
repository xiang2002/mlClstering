import torch
def self_tuning(X, k=10, t_factor=0.5):
    """
    # 基于self-tuning方法构建图的权重矩阵。

    # 参数:
    - X: 输入数据，形状为 (n, d)，其中 n 是样本数，d 是特征数。
    - k: 每个节点的近邻数量。
    - t_factor: 用于控制局部自适应带宽的参数。

    # 返回:
    - W: 图的权重矩阵，形状为 (n, n)。
    """
    n = X.shape[0]

    # 计算欧氏距离的平方矩阵
    dist_matrix = torch.cdist(X, X, p=2) ** 2

    # 为每个点找到k个近邻
    knn_indices = dist_matrix.topk(k + 1, dim=1, largest=False).indices[
        :, 1:
    ]  # 排除自己

    # 初始化权重矩阵
    W = torch.zeros(n, n, device=X.device,dtype=X.dtype)

    # 自适应带宽计算及权重更新
    for i in range(n):
        neighbors = knn_indices[i]

        # 计算局部自适应带宽
        sigma_i = dist_matrix[i, neighbors].mean() * t_factor
        sigma_j = dist_matrix[neighbors, :][:, i].mean() * t_factor

        # 计算高斯核权重并填入权重矩阵
        for j, neighbor in enumerate(neighbors):
            W[i, neighbor] = torch.exp(-dist_matrix[i, neighbor] / (sigma_i * sigma_j))

    # 确保权重矩阵对称
    W = (W + W.T) / 2
#    print(X.dtype,W.dtype)
    return W
