import torch
from KLM.layers import sink_algorithm, arange_like, dual_softmax


def compute_matches(scores, p=0.2):
    # scores: shape [b, m+1, n+1]
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    indices0, indices1 = max0.indices, max1.indices
    mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
    mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
    zero = scores.new_tensor(0)
    # mscores0 = torch.where(mutual0, max0.values.exp(), zero)
    mscores0 = torch.where(mutual0, max0.values, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
    # valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
    valid0 = mutual0 & (mscores0 > p)
    valid1 = mutual1 & valid0.gather(1, indices1)
    indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
    indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

    return indices0, indices1, mscores0, mscores1


def compute_matches_without_mutual(scores, p=0.2):
    # scores: [b, m+1, n+1]
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    indices0, indices1 = max0.indices, max1.indices

    zero = scores.new_tensor(0)

    # Extract scores
    mscores0 = max0.values
    mscores1 = max1.values

    # Set scores to zero where below threshold
    mscores0 = torch.where(mscores0 > p, mscores0, zero)
    mscores1 = torch.where(mscores1 > p, mscores1, zero)

    # Set indices to -1 where scores are below threshold
    indices0 = torch.where(mscores0 > 0, indices0, indices0.new_tensor(-1))
    indices1 = torch.where(mscores1 > 0, indices1, indices1.new_tensor(-1))

    return indices0, indices1, mscores0, mscores1


def compute_score(dist, dustbin, iteration, with_sinkhorn=True):
    if with_sinkhorn:
        score = sink_algorithm(M=dist, dustbin=dustbin,
                               iteration=iteration)  # [nI * nB, N, M]
    else:
        score = dual_softmax(M=dist, dustbin=dustbin)
    return score