import os
import sys
import argparse
import traceback
import yaml
from tqdm import tqdm
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from train import load_network
from components.readers import StandardReader
from components.evaluators import auc_eval
from eval.utils import compute_pose_error, pose_auc, normalize_intrinsic
from eval.matching import matching_iterative, matching_iterative_uncertainty, cal_NN_metric, cal_S_metric
from utils import str2bool
sys.path.append("..")


parser = argparse.ArgumentParser(description='CFM', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--network', type=str, default='cfm')
parser.add_argument('--dataset', type=str, default='yfcc')
parser.add_argument('--feature', type=str, default='spp')
parser.add_argument('--use_iterative', type=str2bool, default=False)
parser.add_argument('--use_uncertainty', type=str2bool, default=False)
parser.add_argument('--use_nn', type=str2bool, default=False)
parser.add_argument('--use_filter', type=str2bool, default=False)
parser.add_argument('--base_dir', type=str, default=".")
parser.add_argument('--weight', type=str, required=True)
parser.add_argument('--num_kpt', type=int, default=2000)

parser.add_argument('--filter_type', type=str, choices=['nn', 'n3net', 'sinkhorn'], default='nn')
parser.add_argument('--valid_layers', nargs='+', type=int, default=[3, 5, 8])
parser.add_argument('--iter_num', type=int, default=1)
parser.add_argument('--sigma_spat', type=float, default=0.2)
parser.add_argument('--use_prune', type=str2bool, default=False)
parser.add_argument('--layer_prune', nargs='+', type=str2bool, default=[True, True])
parser.add_argument('--threshold', nargs='+', type=float, default=[-0.975, -0.8])
parser.add_argument('--n_min_tokens', nargs='+', type=float, default=[0.95, 0.9])
parser.add_argument('--last_sinkhorn', type=str2bool, default=False)
parser.add_argument('--match_threshold', type=float, default=0.2)
parser.add_argument('--filter_thr', type=float, default=0)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--output_file', type=str, default='results.txt')

parser.add_argument('--use_global', type=str2bool, default=True)

parser.add_argument('--use_poselib', type=str2bool, default=False)
parser.add_argument('--error_th', type=float, default=-1)


def evaluation(model):
    use_nn = args.use_nn
    use_filter = args.use_filter
    use_iterative = args.use_iterative
    use_uncertatinty = args.use_uncertainty
    use_poselib = args.use_poselib
    match_thr = args.match_threshold
    filter_thr = args.filter_thr
    nI = matcher_config['layer_num']
    error_th = eval_config['error_th'] if args.error_th != -1 else args.error_th
    stop_criteria = {'match': 0.7, 'pose': 1.5}
    thresholds = [5, 10, 20]
    pose_errors = []
    precisions = []
    matching_scores = []
    total_time = 0
    # count_portion = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}

    for index in tqdm(range(len(reader_loader)), total=len(reader_loader)):
        info = reader.run(index=index)
        img0, img1, K0, K1 = info['img1'], info['img2'], info['K1'], info['K2']
        x0, x1, descs0, descs1 = info['x1'], info['x2'], info['desc1'], info['desc2']
        E, R, t = info['e'], info['r_gt'], info['t_gt']
        pts0, pts1 = x0[:, :2], x1[:, :2]
        scores0, scores1 = x0[:, 2], x1[:, 2]
        norm_pts0 = normalize_intrinsic(x=pts0, K=K0)
        norm_pts1 = normalize_intrinsic(x=pts1, K=K1)
        T_0to1 = np.hstack([R, t.reshape(3, 1)])

        feed_data = {
                    'keypoints0': torch.from_numpy(pts0).cuda().float()[None],
                    'keypoints1': torch.from_numpy(pts1).cuda().float()[None],
                    'keypoints0_3d': torch.from_numpy(norm_pts0).cuda().float()[None],
                    'keypoints1_3d': torch.from_numpy(norm_pts1).cuda().float()[None],
                    'image0': torch.from_numpy(img0).cuda().float().permute(2, 0, 1)[None],
                    'image1': torch.from_numpy(img1).cuda().float().permute(2, 0, 1)[None],
                    'descriptors0': torch.from_numpy(descs0).cuda().float()[None],
                    'descriptors1': torch.from_numpy(descs1).cuda().float()[None],
                    'scores0': torch.from_numpy(scores0).cuda().float()[None],
                    'scores1': torch.from_numpy(scores1).cuda().float()[None],
                    'K0': K0,
                    'K1': K1,
                    'T_0to1': T_0to1,
                    'gt_E': torch.from_numpy(E).cuda().float()[None]
                }

        if use_iterative:
            if use_uncertatinty:
                pred_R, pred_t, precision_, matching_score_ = matching_iterative_uncertainty(
                    nI=nI,
                    data=feed_data,
                    model=model,
                    error_th=error_th,
                    stop_criteria=stop_criteria,
                    match_thr=match_thr,
                    filter_thr=filter_thr,
                    use_nn=use_nn,
                    refine=use_filter
                )
            else:
                pred_R, pred_t, precision_, matching_score_ = matching_iterative(
                    nI=nI,
                    data=feed_data,
                    model=model,
                    error_th=error_th,
                    stop_criteria=stop_criteria,
                    match_thr=match_thr,
                    filter_thr=filter_thr,
                    use_nn=use_nn,
                    refine=use_filter
                )
        else:
            try:
                start_time = time.time()
                match_out = net.produce_matches(data=feed_data)
                end_time = time.time()
                total_time += (end_time - start_time)

                cl_ret = match_out['filter_ret'][-1]
                if use_nn:
                    ret, precision_, matching_score_ = cal_NN_metric(cl_ret, filter_thr, K0, K1, error_th,
                                                                     gt_E=E, num=pts0.shape[0], refine=use_filter,
                                                                     use_poselib=use_poselib)
                else:
                    ret, precision_, matching_score_ = cal_S_metric(cl_ret, filter_thr, match_thr, K0, K1, error_th,
                                                                    gt_E=E, num=pts0.shape[0], refine=use_filter,
                                                                    use_poselib=use_poselib)
                _, pred_R, pred_t, _ = ret
            except Exception:
                traceback.print_exc()
                pred_R, pred_t = None, None
                precision_, matching_score_ = 0, 0

        if pred_R is None:
            err_t, err_R = np.inf, np.inf
            precision, matching_score = 0, 0
        else:
            err_t, err_R = compute_pose_error(T_0to1=T_0to1, R=pred_R, t=pred_t)
            precision, matching_score = precision_, matching_score_

        pose_errors.append(np.max([err_R, err_t]))
        precisions.append(precision)
        matching_scores.append(matching_score)
        aucs = pose_auc(pose_errors, thresholds)
        aucs = [100. * yy for yy in aucs]
        prec = 100. * np.mean(precisions)
        ms = 100. * np.mean(matching_scores)
    with open(args.output_file, 'a') as f:
        f.write(f'Evaluation Results of {args.weight} (mean over {len(pose_errors)} pairs):\n')
        f.write('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t Time\t\n')
        f.write(f'{aucs[0]:.2f}\t {aucs[1]:.2f}\t {aucs[2]:.2f}\t '
                f'{prec:.2f}\t {ms:.2f}\t {total_time}\t {total_time / len(pose_errors)}\n')


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    args.base_dir = Path(args.base_dir)
    if not args.base_dir.exists():
        args.base_dir = Path("/home/u1120220257/hyb/data/mffm/")
    config_path = f'configs/eval_{args.dataset}_{args.feature}.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, yaml.Loader)
        read_config = config['reader']
        read_config['rawdata_dir'] = args.base_dir / read_config['rawdata_dir']
        read_config['dataset_dir'] = args.base_dir / read_config['dataset_dir']
        eval_config = config['evaluator']
        matcher_config = config['matcher']

    read_config['num_kpt'] = args.num_kpt
    if args.num_kpt != 2000:
        read_config['dataset_dir'] = str(read_config['dataset_dir']).replace('2000', f'{args.num_kpt}')
        read_config['dataset_dir'] = Path(read_config['dataset_dir'].replace('mffm/yfcc', f'mffm/yfcc_{args.num_kpt}'))

    print("Number of keypoints:", args.num_kpt)
    print("Dataset Dir:", read_config['dataset_dir'])

    reader = StandardReader(config=read_config)
    reader_loader = DataLoader(dataset=reader, num_workers=4, shuffle=False)
    evaluator = auc_eval(config=eval_config)

    matcher_config['filter'] = {
        'iter_num': args.iter_num,
        'sigma_spat': args.sigma_spat,
        'thr': args.filter_thr,
        'use_global': args.use_global,
    }
    config = {
        'descriptor_dim': 256 if args.feature == 'spp' else 128,
        'GNN_layers': ['self', 'cross'] * matcher_config['layer_num'],
        'match_threshold': args.match_threshold,
        'n_layers': matcher_config['layer_num'],
        'filter': matcher_config['filter'],
        'filter_type': args.filter_type,
        'valid_layers': args.valid_layers,
        'use_prune': args.use_prune,
        'layer_prune': args.layer_prune,
        'threshold': args.threshold,
        'n_min_tokens': args.n_min_tokens,
        'last_sinkhorn': args.last_sinkhorn,
    }

    net = load_network(args.network)(config)
    net.load_state_dict(state_dict=torch.load(args.weight, map_location=torch.device('cpu'))['model'], strict=False)
    net = net.cuda().eval()
    with torch.no_grad():
        evaluation(model=net)
