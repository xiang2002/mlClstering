from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np
def align_clusters(true_labels, predicted_labels):
    """
    对齐聚类结果，使预测的聚类结果与真实标签尽可能匹配
    :param true_labels: 真实标签
    :param predicted_labels: 聚类结果
    :return: 重新编号后的聚类结果
    """
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # 使用匈牙利算法找到最优的标签对齐方式
    row_ind, col_ind = linear_sum_assignment(-cm)  # 最小化代价，所以对混淆矩阵取负
    
    # 创建一个映射，将聚类标签与真实标签对齐
    label_mapping = {col: row for row, col in zip(row_ind, col_ind)}
    
    # 重新编号预测的聚类结果
    aligned_labels = np.array([label_mapping[label] for label in predicted_labels])
    
    return aligned_labels