import torch
import numpy as np
from distances import *
from sinkhornTransport import sinkhornDistance


def TCOT(x: torch.Tensor, y: torch.Tensor, verbose=0, lmbda=1,
          tol=.5e-2, maxIter=100, p_norm='inf'):
    assert y.size(1) != x.size(1), "The dimensions of instances in the input sequences must be the same!"

    N = x.size(0)   # x: [N, d]
    M = y.size(0)   # y: [M, d]
    col_x = torch.arange(1, N + 1) / N
    col_x = col_x.view(N, 1)
    col_y = torch.arange(1, M + 1) / M
    relative_pos = col_x - col_y

    D = x.unsqueeze(1) - y
    D = torch.sum(D**2, dim=-1)
    scale = 1 + torch.abs(relative_pos)
    D = D * scale

    K = torch.exp(-lmbda*D)

    U = K * D

    a = torch.ones(N, 1) / N
    b = torch.ones(M, 1) / M

    dist, lowerEMD, l, m = sinkhornDistance(a, b, K, U, lmbda,
                                            p_norm=p_norm, tol=tol, maxIter=maxIter, verbose=verbose)

    T = m.T * (l * K)   # This is the optimal transport
    return dist, T
