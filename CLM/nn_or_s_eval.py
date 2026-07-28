import torch

from KLM.utils import compute_score, compute_matches_without_mutual
from KLM.layers import arange_like
from CLM.utils import computeNN


def s_eval(data, opt):
    kpts0, kpts1 = data["keypoints0_3d"], data["keypoints1_3d"]  # [b, m, 2]  # [b, n, 2]
    descs0, descs1 = data["mdesc0"], data["mdesc1"]  # [b, d, m]  [b, d, n]

    dist = torch.einsum('bdn,bdm->bnm', descs0, descs1)
    dist = dist / opt['descriptor_dim'] ** .5
    pred_score = compute_score(dist=dist, dustbin=opt['bin_score'].to(dist.device),
                               iteration=opt["sinkhorn_iterations"])
    matches0, matches1, matching_scores0, _ = compute_matches_without_mutual(scores=pred_score, p=0)

    kpt0, kpt1 = kpts0[0], kpts1[0]  # [m, 2] [n, 2]
    idx_sort = matches0[0]  # [m] if kpt0 has match, store the kpt1 index, else -1
    kpt1 = kpt1[idx_sort, :]  # [m, 2]
    x = torch.cat((kpt0, kpt1), dim=-1)  # [m, 4]
    mutual_nearest = arange_like(matches0, 1)[None] == matches1.gather(1, matches0)
    ret_nearest, ret_pred_score = mutual_nearest, pred_score
    if x.shape[0] < 8:
        return None
    xs = x[None][None]  # [b, 1, num, 4]

    logits = [torch.ones((1, xs.shape[2]), device=xs.device)]

    ret = {"xs": xs, "logits": logits,
           "mutual_nearest": ret_nearest, "pred_score": ret_pred_score,
           "matching_score": matching_scores0,
           "masks": None}
    return ret


def nn_eval(data):
    kpts0, kpts1 = data["keypoints0_3d"], data["keypoints1_3d"]  # [b, m, 2]  # [b, n, 2]
    descs0, descs1 = data["mdesc0"], data["mdesc1"]  # [b, d, m]  [b, d, n]
    descs0 = descs0.transpose(1, 2)  # [b, m, d]
    descs1 = descs1.transpose(1, 2)  # [b, n, d]

    kpt0, kpt1 = kpts0[0], kpts1[0]  # [m, 2], [n, 2]
    desc0, desc1 = descs0[0], descs1[0]  # [m, d], [n, d]
    idx_sort_all, _, mutual_nearest, pred_score = computeNN(desc0, desc1, kpt0.device)
    idx_sort = idx_sort_all[1]  # [m]
    kpt1_1 = kpt1[idx_sort, :]  # [m, 2]
    x = torch.cat((kpt0, kpt1_1), dim=-1)  # [m, 4]
    ret_nearest, ret_pred_score = mutual_nearest, pred_score
    if x.shape[0] < 8:
        return None
    xs = x[None][None]  # [b, 1, num, 4]

    logits = [torch.ones((1, xs.shape[2]), device=xs.device)]

    ret = {"xs": xs, "logits": logits,
           "mutual_nearest": ret_nearest, "pred_score": ret_pred_score,
           "masks": None}
    return ret
