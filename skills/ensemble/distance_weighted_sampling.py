import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_distance(x):
    """
    get the Euclidean distance between each pair of points in x

    speeds up the eculidean distance computation using matrix multiplication:
    dist[0, 0] = sqrt((x1-y1)^2 + (x2-y2)^2 + ... + (xn-yn)^2)
               = sqrt((x1^2 + ... + xn^2) + (y1^2 + ... + yn^2) - 2*x.T*y)
               = sqrt(2 - 2*x.T*y))  # because of normalization sums to 1 
    see: https://mlxai.github.io/2017/01/03/finding-distances-between-data-points-with-numpy.html 

    args:
        x: (n_points, n_dims). Each row is normalized already with L2, so they sum to 1
    return:
        dist: (n_points, n_points)
    """
    _x = x.detach()  # (N, *)
    sim = torch.matmul(_x, _x.t())  # (N, N)
    sim = torch.clamp(sim, max=1.0)
    dist = 2 - 2*sim
    # dist += torch.eye(dist.shape[0]).to(dist.device)  # elements have distance 1 to themselves
    dist = dist.sqrt()
    
    return dist


class DistanceWeightedSampling(nn.Module):
    """
    taken from 
    https://github.com/suruoxi/DistanceWeightedSampling/blob/ea8561ad0e6d6e728e3ec121fd8cf500c52f83f8/model.py#L93

    described in paper: Sampling Matters in Deep Embedding Learning
    https://arxiv.org/pdf/1706.07567.pdf

    parameters
    ----------
    batch_k: int
        number of images per class
    nonzero_loss_cutoff:
        default of sqrt(2) as per the sampling paper, which the mean of the gaussian distribution.
        It is said that loss thresholds smaller than this induces no loss, and thus no progress for learning
    Inputs:
        data: input tensor with shapeee (batch_size, edbed_dim)
            Here we assume the consecutive batch_k examples are of the same class.
            For example, if batch_k = 5, the first 5 examples belong to the same class,
            6th-10th examples belong to another class, etc.
    Outputs:
        a_indices: indicess of anchors
        x[a_indices]
        x[p_indices]
        x[n_indices]
        xxx
    """
    def __init__(self, 
                cutoff=1e-8, 
                nonzero_loss_cutoff=1.4, 
                normalize=True, 
                **kwargs):
        super(DistanceWeightedSampling, self).__init__()
        self.cutoff = cutoff
        self.nonzero_loss_cutoff = nonzero_loss_cutoff
        self.normalize = normalize

    def forward(self, x):
        n, d = x.shape  # (N, num_modules * embedding_size)
        x = F.normalize(x)
        distance = get_distance(x)
        distance = distance.clamp(min=self.cutoff)  # minimum distance is self.cutoff

        # embeddings are constrained to the n-dimensional unit sphere (because each data point is normalized) for large n>128
        # assume the points are uniformly distributed on the sphere
        # the following is the inverse of the uniform pair-wise distance on a sphere
        negative_log_weights = ((2.0 - float(d)) * distance.log() - (float(d-3)/2)*torch.log(torch.clamp(1.0 - 0.25*(distance*distance), min=1e-8)))
        positive_log_weights = -negative_log_weights
        if self.normalize:
            negative_log_weights = (negative_log_weights - negative_log_weights.min()) / (negative_log_weights.max() - negative_log_weights.min() + 1e-8)
            positive_log_weights = (positive_log_weights - positive_log_weights.min()) / (positive_log_weights.max() - positive_log_weights.min() + 1e-8)

        negative_weights = torch.exp(negative_log_weights - torch.max(negative_log_weights))
        positive_weights = torch.exp(positive_log_weights - torch.max(positive_log_weights))

        mask = torch.ones_like(negative_weights) - torch.eye(n).to(x.device)
        mask_uniform_probs = mask / (n-1)  # normalize row sum to 1

        negative_weights = negative_weights * mask * ((distance < self.nonzero_loss_cutoff).float()) + 1e-8
        negative_weights = negative_weights / torch.sum(negative_weights, dim=1, keepdim=True)  # normalize row sum to 1
        positive_weights = positive_weights * mask + 1e-8
        positive_weights = positive_weights / torch.sum(positive_weights, dim=1, keepdim=True)  # normalize row sum to 1

        positive_indices = torch.argmax(positive_weights, dim=1)
        negative_indices = torch.argmax(negative_weights, dim=1)

        class_label_matrix = torch.ones_like(distance).to(x.device)
        class_label_matrix[torch.arange(n).to(x.device), negative_indices] = 0

        return x, class_label_matrix
