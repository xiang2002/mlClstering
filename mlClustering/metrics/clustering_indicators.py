def indicators (pred, data=None, labels=None, model_name='cluster', verbose=1):
    '''
    计算聚类指标
    :param pred: 聚类结果
    :param data: 原始数据
    :param labels: 真实聚类标签
    :param model_name: 打印的模型名称
    :param verbose: 是否打印
    :return: 聚类各指标的dict
    '''
    measure_dict = dict()
    #如果有原始数据
    if data is not None:
        measure_dict['si'] = SI(data, pred)
        measure_dict['ch'] = CH(data, pred)
       #如果数据有标签
    if labels is not None:
        measure_dict['acc'] = cluster_acc(pred, labels)[0]
        measure_dict['nmi'] = NMI(labels, pred)
        measure_dict['ar'] = AR(labels, pred)
        measure_dict['ami'] = AMI(labels, pred)

#如果需要打印所有指标
    if verbose:
        char = ''
        for (key, value) in measure_dict.items():
            char += '{}: {:.4f} '.format(key, value)
        print('{} {}'.format(model_name, char))

    return measure_dict

##参考论文Unsupervised deep embedding for clustering analysis
def cluster_acc(Y_pred, Y):
    '''
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        Y_pred: Numpy array. Predicted labels
        Y: Numpy array. True labels
    # Return
        accuracy: ACC value
    '''
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    total = 0
    for i in range(len(ind[0])):
        total += w[ind[0][i], ind[1][i]]
    return total * 1.0 / Y_pred.size, w