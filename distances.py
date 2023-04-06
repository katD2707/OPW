import torch


def pdist2(X, Y, metric='sqreuclidean'):
    if metric.lower() == 'sqreuclidean':
        return distEucSqr(X, Y)
    elif metric.lower() == 'euclidean':
        return torch.sqrt(distEucSqr(X, Y))
    elif metric.lower() == 'L1':
        return distL1(X, Y)
    elif metric.lower() == 'cosine':
        return distCosine(X, Y)
    elif metric.lower() == 'emd':
        return distEmd(X, Y)
    elif metric.lower() == 'chisqr':
        return distChiSqr(X, Y)
    else:
        raise NotImplementedError(f'pdist - unknown metric: {metric}')


def distL1(x: torch.Tensor, y: torch.Tensor):
    return torch.abs(x.unsqueeze(1) - y).sum(dim=-1)


def distCosine(x: torch.Tensor, y: torch.Tensor, eps=1e-8):
    assert x.dtype == torch.float or y.dtype == torch.float, "Inputs must be of type float"
    cos = torch.nn.CosineSimilarity(dim=-1, eps=eps)
    return 1 - cos(x.unsqueeze(1), y)


def distEmd(x: torch.Tensor, y: torch.Tensor):
    x_cdf = torch.cumsum(x, dim=-1)
    y_cdf = torch.cumsum(y, dim=-1)

    return torch.abs(x_cdf.unsqueeze(1) - y_cdf).sum(dim=-1)


def distEucSqr(x: torch.Tensor, y: torch.Tensor):
    return torch.cdist(x, y, p=2)**2


def distChiSqr(x: torch.Tensor, y: torch.Tensor, eps=1e-10):
    a = x.unsqueeze(1) + y
    b = x.unsqueeze(1) - y
    return (b**2 / (a + eps)).sum(dim=-1) / 2

