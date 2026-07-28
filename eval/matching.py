import cv2
import numpy as np
from KLM.utils import *
from components.utils.metrics import compute_epi_inlier
from eval.pose_estimation import estimate_pose, estimate_pose_poselib
from eval.utils import angle_error_mat, angle_error_vec, denormalize_intrinsic


def cal_related_metric(mkpts0, mkpts1, K0, K1, error_th, norm_mkpts0, norm_mkpts1, gt_E, num, use_poselib):
    if use_poselib:
        ret = estimate_pose_poselib(mkpts0, mkpts1, K0, K1, 2.0)
    else:
        ret = estimate_pose(mkpts0, mkpts1, K0, K1, error_th, method=cv2.USAC_MAGSAC)
    correct, epi_errs = compute_epi_inlier(x1=norm_mkpts0, x2=norm_mkpts1, E=gt_E, inlier_th=0.005, return_error=True)
    num_correct = np.sum(correct)
    precision = np.mean(correct) if len(correct) > 0 else 0
    matching_score = num_correct / num if num > 0 else 0

    return ret, precision, matching_score


def cal_S_metric(cl_ret, filter_thr, match_thr, K0, K1, error_th, gt_E=None, num=1024, refine=True, use_poselib=False):
    xs = cl_ret['xs'].squeeze()  # [n, 4]
    logits = cl_ret['logits'][-1].squeeze()  # [n]
    filter_scores = cl_ret['matching_score'].squeeze()  # [n]
    mutual_nearest = cl_ret['mutual_nearest'].squeeze()  # [n]
    if mutual_nearest is not None and mutual_nearest.shape[0] == logits.shape[0]:
        xs = xs[mutual_nearest]  # [n1, 4]
        logits = logits[mutual_nearest]  # [n1]
        filter_scores = filter_scores[mutual_nearest]  # [n1]
    norm_mkpts0, norm_mkpts1 = xs[:, :2].cpu().detach().numpy(), xs[:, 2:].cpu().detach().numpy()
    mkpts0, mkpts1 = denormalize_intrinsic(x=norm_mkpts0, K=K0), denormalize_intrinsic(x=norm_mkpts1, K=K1)

    if refine:
        weights = torch.relu(torch.tanh(logits))
        mask = (weights.cpu().detach().numpy() > filter_thr) & (filter_scores.cpu().detach().numpy() > match_thr)
        mkpts0, mkpts1 = mkpts0[mask], mkpts1[mask]
        norm_mkpts0, norm_mkpts1 = norm_mkpts0[mask], norm_mkpts1[mask]

    return cal_related_metric(mkpts0, mkpts1, K0, K1, error_th, norm_mkpts0, norm_mkpts1, gt_E, num, use_poselib)


def cal_NN_metric(cl_ret, filter_thr, K0, K1, error_th, gt_E=None, num=1024, refine=True, use_poselib=False):
    xs = cl_ret['xs'].squeeze()  # [n, 4]
    logits = cl_ret['logits'][-1].squeeze()  # [n]
    mutual_nearest = cl_ret['mutual_nearest'].squeeze() if 'mutual_nearest' in cl_ret else None  # [n]
    y_hat = cl_ret.get('y_hat')
    if mutual_nearest is not None and mutual_nearest.shape[0] == logits.shape[0]:
        xs = xs[mutual_nearest]  # [n1, 4]
        logits = logits[mutual_nearest]  # [n1]
    norm_mkpts0, norm_mkpts1 = xs[:, :2].cpu().detach().numpy(), xs[:, 2:].cpu().detach().numpy()
    mkpts0, mkpts1 = denormalize_intrinsic(x=norm_mkpts0, K=K0), denormalize_intrinsic(x=norm_mkpts1, K=K1)

    if refine:
        if y_hat is None:
            weights = torch.relu(torch.tanh(logits))
            mask = weights.cpu().detach().numpy() > filter_thr
        else:
            mask = y_hat.cpu().detach().numpy() < filter_thr
        mkpts0, mkpts1 = mkpts0[mask], mkpts1[mask]
        norm_mkpts0, norm_mkpts1 = norm_mkpts0[mask], norm_mkpts1[mask]

    return cal_related_metric(mkpts0, mkpts1, K0, K1, error_th, norm_mkpts0, norm_mkpts1, gt_E, num, use_poselib)


def matching_iterative(data, model, nI, error_th, stop_criteria, match_thr, filter_thr, use_nn=True, refine=True):
    pts0, pts1 = data['keypoints0'], data['keypoints1']  # [b, m, 2]  [b, n, 2]
    norm_kpts0, norm_kpts1 = data['keypoints0_3d'], data['keypoints1_3d']  # [b, m, 2]  [b, n, 2]
    scores0, scores1 = data['scores0'], data['scores1']
    desc0, desc1 = data['descriptors0'], data['descriptors1']
    desc0, desc1 = desc0.transpose(1, 2), desc1.transpose(1, 2)
    K0, K1 = data['K0'], data['K1']
    E = data['gt_E'][0].cpu().detach().numpy()  # [b, 3, 3]
    last_best_R = None
    last_best_t = None
    precision, matching_score = 0, 0
    # valid_its = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    valid_its = [3, 5, 7, 8]

    for it in range(nI):
        precision_it, matching_score_it = 0, 0
        if it == 0:
            enc0, enc1 = model.encode_keypoint(norm_kpts0=norm_kpts0,
                                               norm_kpts1=norm_kpts1,
                                               scores0=scores0,
                                               scores1=scores1)
            desc0 = desc0 + enc0
            desc1 = desc1 + enc1

        try:
            desc0, desc1 = model.forward_one_layer(desc0=desc0, desc1=desc1, layer_i=it * 2, data=data)
            desc0, desc1, cl_ret = model.forward_one_layer(desc0=desc0, desc1=desc1, layer_i=it * 2 + 1, data=data)
        except Exception:
            continue
            
        if it not in valid_its:
            continue

        if cl_ret is not None:
            if use_nn:
                ret, precision_it, matching_score_it = cal_NN_metric(cl_ret, filter_thr, K0, K1, error_th,
                                                                     gt_E=E, num=pts0.shape[1], refine=refine)
            else:
                ret, precision_it, matching_score_it = cal_S_metric(cl_ret, filter_thr, match_thr, K0, K1, error_th,
                                                                    gt_E=E, num=pts0.shape[1], refine=refine)
        else:
            ret = None

        if ret is not None:
            _, R, t, _ = ret
        else:
            R, t = None, None
        if it >= 1:
            diff_R = angle_error_mat(R1=last_best_R, R2=R) if last_best_R is not None and R is not None else np.inf
            diff_t = angle_error_vec(v1=last_best_t, v2=t) if last_best_t is not None and t is not None else np.inf
        else:
            diff_R, diff_t = np.inf, np.inf

        pose_diff = np.max([diff_R, diff_t])
        last_best_R = R
        last_best_t = t
        precision = precision_it
        matching_score = matching_score_it

        # Check if stop iteration
        if 'pose' in stop_criteria.keys():
            if pose_diff <= stop_criteria['pose']:
                # print("stop at iteration:", it)
                return R, t, precision_it, matching_score_it

    return last_best_R, last_best_t, precision, matching_score


def matching_iterative_uncertainty(data, model, nI, error_th, stop_criteria, match_thr, filter_thr, use_nn=True, refine=True):
    pts0, pts1 = data['keypoints0'], data['keypoints1']  # [b, m, 2]  [b, n, 2]
    norm_kpts0, norm_kpts1 = data['keypoints0_3d'], data['keypoints1_3d']  # [b, m, 2]  [b, n, 2]
    scores0, scores1 = data['scores0'], data['scores1']
    desc0, desc1 = data['descriptors0'], data['descriptors1']
    desc0, desc1 = desc0.transpose(1, 2), desc1.transpose(1, 2)
    K0, K1 = data['K0'], data['K1']
    E = data['gt_E'][0].cpu().detach().numpy()  # [b, 3, 3]
    last_best_R = None
    last_best_t = None
    precision, matching_score = 0, 0
    # valid_its = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    valid_its = [3, 5, 7, 8]

    sel_ids0 = None
    sel_ids1 = None
    update_0 = False
    update_1 = False
    enc0, enc1 = model.encode_keypoint(norm_kpts0=norm_kpts0,
                                       norm_kpts1=norm_kpts1,
                                       scores0=scores0,
                                       scores1=scores1)
    desc0 = desc0 + enc0
    desc1 = desc1 + enc1

    for it in range(nI):
        precision_it, matching_score_it = 0, 0
        if update_0:
            desc0 = desc0[:, :, sel_ids0]
            data['keypoints0_3d'] = data['keypoints0_3d'][:, sel_ids0, :]
            data['scores0'] = data['scores0'][:, sel_ids0]
            norm_kpts0 = norm_kpts0[:, sel_ids0, :]

        if update_1:
            desc1 = desc1[:, :, sel_ids1]
            data['keypoints1_3d'] = data['keypoints1_3d'][:, sel_ids1, :]
            data['scores1'] = data['scores1'][:, sel_ids1]
            norm_kpts1 = norm_kpts1[:, sel_ids1, :]

        try:
            desc0, desc1 = model.forward_one_layer(desc0=desc0, desc1=desc1, layer_i=it * 2, data=data)
            desc0, desc1, cl_ret = model.forward_one_layer(desc0=desc0, desc1=desc1, layer_i=it * 2 + 1, data=data)
        except Exception:
            continue

        if it not in valid_its:
            update_0 = False
            update_1 = False
            continue

        prob00 = model.self_prob0
        prob11 = model.self_prob1
        prob01 = model.cross_prob0
        prob10 = model.cross_prob1
        if use_nn:
            pred_score = cl_ret['pred_score']
        else:
            pred_score = cl_ret['pred_score'][:, :-1, :-1]

        if cl_ret is not None:
            if use_nn:
                ret, precision_it, matching_score_it = cal_NN_metric(cl_ret, filter_thr, K0, K1, error_th,
                                                                     gt_E=E, num=pts0.shape[1], refine=refine)
            else:
                ret, precision_it, matching_score_it = cal_S_metric(cl_ret, filter_thr, match_thr, K0, K1, error_th,
                                                                    gt_E=E, num=pts0.shape[1], refine=refine)
        else:
            ret = None

        if ret is not None:
            _, R, t, inliers = ret
            pose_inliers = inliers
            inlier_ratio = np.sum(pose_inliers) / norm_kpts0.shape[1]
        else:
            R, t = None, None
            inlier_ratio = 0
        if it >= 1:
            diff_R = angle_error_mat(R1=last_best_R, R2=R) if last_best_R is not None and R is not None else np.inf
            diff_t = angle_error_vec(v1=last_best_t, v2=t) if last_best_t is not None and t is not None else np.inf
        else:
            diff_R, diff_t = np.inf, np.inf

        pose_diff = np.max([diff_R, diff_t])
        last_best_R = R
        last_best_t = t
        precision = precision_it
        matching_score = matching_score_it

        # performing adaptive pooling
        if inlier_ratio == 0:
            mscore_th = 0.2
        else:
            mscore_th = 0.2 * inlier_ratio

        sel_ids0, sel_ids1 = model.pool(pred_score=pred_score, prob00=prob00, prob01=prob01, prob11=prob11,
                                        prob10=prob10, mscore_th=mscore_th, uncertainty_ratio=1.0, use_nn=use_nn)
        update_0 = False if sel_ids0 is None else True
        update_1 = False if sel_ids1 is None else True

        # Check if stop iteration
        if 'pose' in stop_criteria.keys():
            if pose_diff <= stop_criteria['pose']:
                # print("stop at iteration:", it)
                return R, t, precision_it, matching_score_it

    return last_best_R, last_best_t, precision, matching_score

