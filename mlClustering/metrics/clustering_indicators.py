from sklearn.metrics.cluster import (
    normalized_mutual_info_score as NMI,
    adjusted_mutual_info_score as AMI,
    adjusted_rand_score as AR,
    silhouette_score as SI,
    calinski_harabasz_score as CH,
)

from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np


def indicators(
    pred: np.array,
    data: np.array = None,
    labels: np.array = None,
    model_name: str = "cluster",
    verbose: int = 1,
) -> dict:
    """
    计算聚类指标
    :param pred: 聚类结果
    :param data: 原始数据
    :param labels: 真实聚类标签
    :param model_name: 打印的模型名称
    :param verbose: 是否打印
    :return: 聚类各指标的dict
    """
    measure_dict = dict()
    # 如果有原始数据
    if data is not None:
        measure_dict["si"] = SI(data, pred)
        measure_dict["ch"] = CH(data, pred)
    # 如果数据有标签
    if labels is not None:
        measure_dict["acc"] = cluster_acc(pred, labels)[0]
        measure_dict["nmi"] = NMI(labels, pred)
        measure_dict["ar"] = AR(labels, pred)
        measure_dict["ami"] = AMI(labels, pred)

    # 如果需要打印所有指标
    if verbose:
        char = ""
        for key, value in measure_dict.items():
            char += "{}: {:.4f} ".format(key, value)
        print("{} {}".format(model_name, char))

    return measure_dict


##参考论文Unsupervised deep embedding for clustering analysis
from typing import Tuple


def cluster_acc(Y_pred: np.array, Y: np.array) -> Tuple[float, np.array]:
    """
    Calculate clustering accuracy. Require scikit-learn installed.

    参考论文《Unsupervised deep embedding for clustering analysis》

    Arguments:
        Y_pred (np.array): Predicted labels.
        Y (np.array): True labels.

    Returns:
        Tuple[float, np.array]:
            - accuracy (float): ACC value.
            - w (np.array): The confusion matrix.
    """
    # 确保预测标签和真实标签的大小相同
    assert Y_pred.size == Y.size

    # 找到预测标签和真实标签中的最大值，并加1，确定混淆矩阵的维度
    D = max(Y_pred.max(), Y.max()) + 1

    # 初始化一个D x D的零矩阵，用于存储混淆矩阵
    w = np.zeros((D, D), dtype=np.int64)

    # 填充混淆矩阵
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1

    # 使用线性分配算法（匈牙利算法）找到最大匹配
    ind = linear_assignment(w.max() - w)

    # 计算总的匹配数
    total = 0
    for i in range(len(ind[0])):
        total += w[ind[0][i], ind[1][i]]

    # 返回准确率和混淆矩阵
    return total * 1.0 / Y_pred.size, w
