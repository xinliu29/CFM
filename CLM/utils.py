import cv2
import torch
import numpy as np


def np_skew_symmetric(v):
    zero = np.zeros_like(v[:, 0])
    M = np.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], axis=1)
    return M


def batch_episym(x1, x2, F):
    batch_size, num_pts = x1.shape[0], x1.shape[1]
    x1 = torch.cat([x1, x1.new_ones(batch_size, num_pts, 1)], dim=-1).reshape(batch_size, num_pts, 3, 1)
    x2 = torch.cat([x2, x2.new_ones(batch_size, num_pts, 1)], dim=-1).reshape(batch_size, num_pts, 3, 1)
    F = F.reshape(-1, 1, 3, 3).repeat(1, num_pts, 1, 1)
    x2Fx1 = torch.matmul(x2.transpose(2, 3), torch.matmul(F, x1)).reshape(batch_size, num_pts)
    Fx1 = torch.matmul(F, x1).reshape(batch_size, num_pts, 3)
    Ftx2 = torch.matmul(F.transpose(2, 3), x2).reshape(batch_size, num_pts, 3)

    ys = x2Fx1 ** 2 * (
            1.0 / (Fx1[:, :, 0] ** 2 + Fx1[:, :, 1] ** 2 + 1e-15) +
            1.0 / (Ftx2[:, :, 0] ** 2 + Ftx2[:, :, 1] ** 2 + 1e-15))

    return ys


def correctMatches(e_gt):
    step = 0.1
    xx, yy = np.meshgrid(np.arange(-1, 1, step), np.arange(-1, 1, step))
    # Points in first image before projection
    pts1_virt_b = np.float32(np.vstack((xx.flatten(), yy.flatten())).T)
    # Points in second image before projection
    pts2_virt_b = np.float32(pts1_virt_b)
    pts1_virt_b, pts2_virt_b = pts1_virt_b.reshape(1, -1, 2), pts2_virt_b.reshape(1, -1, 2)
    pts1_virt_b, pts2_virt_b = cv2.correctMatches(e_gt.reshape(3, 3), pts1_virt_b, pts2_virt_b)
    return pts1_virt_b.squeeze(), pts2_virt_b.squeeze()


def gtMatches(R, t):
    B = R.shape[0]
    pts_virts = torch.zeros(B, 400, 4, device=R.device)
    for b in range(B):
        R_ = R[b].detach().cpu().numpy()
        t_ = t[b].detach().cpu().numpy()
        e_gt_unnorm = np.reshape(
            np.matmul(
                np.reshape(
                    np_skew_symmetric(t_.astype('float64').reshape(1, 3)), (3, 3)),
                np.reshape(R_.astype('float64'), (3, 3))), (3, 3))
        e_gt = e_gt_unnorm / np.linalg.norm(e_gt_unnorm)
        pts1_virt, pts2_virt = correctMatches(e_gt)  # (400, 2)
        pts_virt = np.concatenate([pts1_virt, pts2_virt], axis=1).astype('float64')  # (400, 4)
        pts_virt = torch.from_numpy(np.stack(pts_virt)).float()  # [400, 4]
        pts_virts[b] = pts_virt
    return pts_virts


def computeNN(desc_ii, desc_jj, device):
    # [m, d], [n, d]
    d1 = (desc_ii ** 2).sum(1)
    d2 = (desc_jj ** 2).sum(1)
    distmat = (d1.unsqueeze(1) + d2.unsqueeze(0) - 2 * torch.matmul(desc_ii, desc_jj.transpose(0, 1))).sqrt()

    pred_score = - distmat
    min_val = torch.min(pred_score)
    max_val = torch.max(pred_score)
    pred_score = (pred_score - min_val) / (max_val - min_val)

    distVals, nnIdx1 = torch.topk(distmat, k=2, dim=1, largest=False)  # [1024, 2] [1024, 2]
    nnIdx11 = nnIdx1[:, 0]  # [1024]
    nnIdx12 = nnIdx1[:, 1]  # [1024]
    _, nnIdx2 = torch.topk(distmat, k=1, dim=0, largest=False)  # [1, 1024]
    nnIdx2 = nnIdx2.squeeze()  # [1024]
    mutual_nearest = (nnIdx2[nnIdx11] == torch.arange(nnIdx11.shape[0]).to(device))  # [1024]
    ratio_test = (distVals[:, 0] / distVals[:, 1].clamp(min=1e-10))  # [1024]
    idx_sort = [torch.arange(nnIdx11.shape[0]), nnIdx11, nnIdx12]
    return idx_sort, ratio_test, mutual_nearest, pred_score[None]


def scores_0_to_1(scores, indexes, n2):
    b, c, n1 = scores.shape  # Extracting dimensions from the scores tensor

    # Create an empty tensor to store the result
    result = torch.zeros((b*c, n2), dtype=scores.dtype, device=scores.device)

    # Create a count tensor to keep track of the number of additions at each position
    count = torch.zeros((b*c, n2), dtype=scores.dtype, device=scores.device)

    # Iterate over each batch and update the result tensor based on indexes
    for i in range(b):
        # Get the indexes for the current batch
        index = indexes[i].to(torch.int64)

        # Update the result tensor and count tensor using scatter_add_
        result[i * c: (i + 1) * c].scatter_add_(1, index.unsqueeze(0).expand(c, n1), scores[i])

        # Increment count tensor for the corresponding positions
        count[i * c: (i + 1) * c].scatter_add_(1, index.unsqueeze(0).expand(c, n1), torch.ones_like(scores[i]))

    # Avoid division by zero
    count[count == 0] = 1

    # Average the result tensor by dividing by the count tensor
    scores_averaged = result / count

    # Reshape back to the original shape [b, c, n]
    scores_averaged = scores_averaged.view(b, c, n2)

    return scores_averaged
