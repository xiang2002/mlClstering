import torch


def CDKM(X, label, c, max_iter=100):
    """
    Input
    X: d*n data (torch.Tensor)(输入数据X实际一般是n*d，这里为了和原代码保持一致，所以第一步先转置了一下)
    label: initial label n*1 (torch.Tensor)
    c: number of clusters (int)
    max_iter: maximum number of iterations (int)

    Output
    Y_label: label vector n*1 (torch.Tensor)
    iter_num: number of iterations (int)
    """  # obj_max: objective function value (max) (list)
    X = X.T  # 转置X为d*n
    d, n = X.shape
    Y = torch.zeros(n, c, dtype=X.dtype, device=X.device)
    Y[torch.arange(n), label] = 1  # transform label into indicator matrix
    last = torch.zeros_like(label, device=X.device)
    iter_num = 0

    # Store once
    aa = Y.sum(dim=0)
    _, label = torch.max(Y, dim=1)
    BBB = 2 * (X.T @ X)
    XX = torch.diag(BBB) / 2

    BBUU = BBB @ Y
    ybby = torch.diag(Y.T @ BBUU / 2)

    # Compute initial objective function value
    # obj_max = [ybby.sum().item() / aa.sum().item()]

    while not torch.equal(label, last) and iter_num < max_iter:
        last = label.clone()
        for i in range(n):
            m = label[i].item()
            if aa[m] == 1:
                continue

            V21 = ybby + (BBUU[i, :] + XX[i]) * (1 - Y[i, :])
            V11 = ybby - (BBUU[i, :] - XX[i]) * Y[i, :]
            delta = V21 / (aa + 1 - Y[i, :]) - V11 / (aa - Y[i, :])
            q = torch.argmax(delta).item()

            if m != q:
                aa[q] += 1
                aa[m] -= 1
                ybby[m] = V11[m]
                ybby[q] = V21[q]
                Y[i, m] = 0
                Y[i, q] = 1
                label[i] = q
                BBUU[:, m] -= BBB[:, i]
                BBUU[:, q] += BBB[:, i]

        iter_num += 1
        # obj_max.append(ybby.sum().item() / aa.sum().item())

    return label, iter_num  # , obj_max
