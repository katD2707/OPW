import torch
import numpy as np
from distances import *


def OPW_w(x: torch.Tensor, y: torch.Tensor, a: torch.Tensor = None, b: torch.Tensor = None, std=1, verbose=0, lambda1=50, lambda2=0.1,
          tol=.5e-2, maxIter=20, p_norm='inf'):
    assert y.size(1) != x.size(1), "The dimensions of instances in the input sequences must be the same!"

    N = x.size(0)
    M = y.size(0)
    col_x = torch.arange(1, N+1)/N
    col_x = col_x.view(N, 1)
    col_y = torch.arange(1, M+1)/M
    relative_pos = col_x-col_y

    l = torch.abs(relative_pos) / ((1/N**2 + 1/M**2)**0.5)
    P = torch.exp(-l**2/(2*std**2)) / (std*(2*np.pi)**0.5)

    S = lambda1 / (relative_pos**2 + 1)

    D = pdist2(x, y, 'sqreuclidean')

    K = P * torch.exp((S - D) / lambda2)

    if a is None:
        a = torch.ones(N, 1) / N

    if b is None:
        b = torch.ones(M, 1) / M

    ainvK = K / a   # [N, M]

    iter = 0
    u = torch.ones(N, 1) / N

    while iter < maxIter:
        u = 1. / torch.matmul(ainvK, (b / (torch.matmul(K.T, u))))
        iter += 1
        if iter % 20 == 1 or iter == maxIter:
            v = b / torch.matmul(K.T, u)    # [M, 1]
            u = 1 / torch.matmul(ainvK, v)  # [N, 1]

            criterion = torch.sum(torch.abs(v * torch.matmul(K.T, u) - b), dim=0)
            criterion = criterion.norm(p=float(p_norm))
            if abs(criterion) < tol:
                break

            iter += 1
            if verbose > 0:
                print(f"Iteration : {iter}, Criterion: {criterion}")

    U = K * D   # [N, M]
    dist = torch.sum(u * torch.matmul(U, v), dim=0)
    T = v.T * (u * K)

    return dist, T