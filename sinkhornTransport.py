import numpy as np
import torch


def sinkhornDistance(a: torch.Tensor, b: torch.Tensor, K, U, lmbda, stopCriterion='marginalDifference',
                     p_norm='inf', tol=.5e-2, maxIter=5000, verbose=0):
    if a.size(1) == 1:
        ONE_VS_N = True
    elif a.size(1) == b.size(1):
        ONE_VS_N = False
    else:
        raise ValueError('The first parameter a is either a column vector in the probability simplex, ' \
              'or N column vectors in the probability simplex where N is size(b,2)')

    if b.size(1) > b.size(0):
        BIGN = True
    else:
        BIGN = False

    if ONE_VS_N:
        someZeroValues = False
        I = a > 0
        if sum(I) < len(I):
            someZeroValues = True
            K = K[I, :]
            U = U[I, :]
            a = a[I]
        ainvK = K / a

    iter = 0
    u = torch.ones(a.size(0), b.size(1))/a.size(0)

    if stopCriterion == 'distanceRelativeDecrease':
        D_old = torch.ones(1, b.size(1))

    while iter < maxIter:
        if ONE_VS_N:
            if BIGN:
                u = 1 / torch.matmul(ainvK, b / torch.matmul(K.T, u))
            else:
                u = 1 / torch.matmul(ainvK, b / torch.matmul(u.T, K).T)
        else:
            if BIGN:
                u = a / torch.matmul(K, b / torch.matmul(u.T, K).T)
            else:
                u = a / torch.matmul(K, b / torch.matmul(K.T, u))

        iter += 1
        if iter % 20 == 1 or iter == maxIter:
            if BIGN:
                v = b / torch.matmul(K.T, u)
            else:
                v = b / torch.matmul(u.T, K).T

            if ONE_VS_N:
                u = 1 / torch.matmul(ainvK, v)
            else:
                u = a / torch.matmul(K, v)

            if stopCriterion == "distanceRelativeDecrease":
                D = torch.sum(u * torch.matmul(U, v))
                criterion = (D/D_old - 1).norm(p=float(p_norm))
                if criterion < tol:
                    break
                D_old = D
            elif stopCriterion == "marginalDifference":
                criterion = torch.sum(torch.abs(v * torch.matmul(K.T, u) - b)).norm(p=float(p_norm))
                if criterion < tol:
                    break
            else:
                raise NotImplementedError('Stopping Criterion not recognized')

            iter += 1
            if verbose > 0:
                print(f'Iteration : {iter}, Criterion: {criterion}')
            if any(np.isnan(criterion)):
                raise OverflowError('NaN values have appeared during the fixed point iteration. This problem appears '
                                    'because of insufficient machine precision when processing computations with a '
                                    'regularization value of lambda that is too high. Try again with a reduced '
                                    'regularization parameter lambda or with a thresholded metric matrix M.')
    if stopCriterion == "marginalDifference":
        D = torch.sum(u * torch.matmul(U, v))

    alpha = torch.log(u)
    beta = torch.log(v)
    beta[beta < -1e8] = 0
    if ONE_VS_N:
        L = (torch.matmul(a.T, alpha) + torch.sum(b * beta)) / lmbda
    else:
        alpha[alpha < -1e8] = 0
        L = (torch.sum(a * alpha) + torch.sum(b * beta)) / lmbda

    if ONE_VS_N:
        if someZeroValues:
            uu = u
            u = torch.zeros(len(I), b.size(1))
            u[I, :] = uu

    return D, L, u, v

