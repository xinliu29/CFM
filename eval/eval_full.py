import yaml
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from components.readers import StandardReader
from components.evaluators import auc_eval
from eval.utils import normalize_intrinsic, denormalize_intrinsic


def feed_match_v2(info, model, match_thr=0.2, filter_thr=0):
    with torch.no_grad():
        img0, img1, K0, K1 = info['img1'], info['img2'], info['K1'], info['K2']
        x0, x1, desc0, desc1 = info['x1'], info['x2'], info['desc1'], info['desc2']
        E = info['e']

        pts0, pts1 = x0[:, :2], x1[:, :2]
        scores0, scores1 = x0[:, 2], x1[:, 2]
        norm_x0 = normalize_intrinsic(x=pts0, K=K0)
        norm_x1 = normalize_intrinsic(x=pts1, K=K1)
        feed_data = {'keypoints0': torch.from_numpy(pts0).cuda().float()[None],
                     'keypoints1': torch.from_numpy(pts1).cuda().float()[None],
                     'keypoints0_3d': torch.from_numpy(norm_x0).cuda().float()[None],
                     'keypoints1_3d': torch.from_numpy(norm_x1).cuda().float()[None],
                     'descriptors0': torch.from_numpy(desc0).cuda().float()[None],
                     'descriptors1': torch.from_numpy(desc1).cuda().float()[None],
                     'scores0': torch.from_numpy(scores0).cuda().float()[None],
                     'scores1': torch.from_numpy(scores1).cuda().float()[None],
                     'gt_E': torch.from_numpy(E).cuda().float()[None]
                     }

        match_out = model.module.produce_matches(data=feed_data, p=match_thr, only_last=True)

        cl_ret = match_out['filter_ret'][-1]
        xs = cl_ret['xs'].squeeze()  # [n, 4]
        logits = cl_ret['logits'][-1].squeeze()  # [n]

        if cl_ret['mutual_nearest'] is not None:
            mutual_nearest = cl_ret['mutual_nearest'].squeeze()
        else:
            mutual_nearest = None

        if cl_ret["masks"] is not None:
            prune_mask = cl_ret['masks'].squeeze()
        else:
            prune_mask = None

        if mutual_nearest is not None and prune_mask is not None:
            final_mask = mutual_nearest * prune_mask
        elif mutual_nearest is not None:
            final_mask = mutual_nearest
        elif prune_mask is not None:
            final_mask = prune_mask
        else:
            final_mask = None

        if final_mask is not None and final_mask.shape[0] == logits.shape[0]:
            xs = xs[final_mask.bool()]  # [n1, 4]
            logits = logits[final_mask.bool()]  # [n1]

        norm_mkpts0 = xs[:, :2].cpu().detach().numpy()  # [n1, 2]
        norm_mkpts1 = xs[:, 2:].cpu().detach().numpy()  # [n1, 2]
        mkpts0 = denormalize_intrinsic(x=norm_mkpts0, K=K0)  # [n1, 2]
        mkpts1 = denormalize_intrinsic(x=norm_mkpts1, K=K1)  # [n1, 2]

        weights = torch.relu(torch.tanh(logits))
        filter_scores = cl_ret['matching_score'].squeeze() if 'matching_score' in cl_ret.keys() else None
        if filter_scores is None:
            mask = weights.cpu().detach().numpy() > filter_thr
        else:
            filter_scores = filter_scores[final_mask]  # n1
            mask = (weights.cpu().detach().numpy() > filter_thr) & (filter_scores.cpu().detach().numpy() > match_thr)
        mkpts0, mkpts1 = mkpts0[mask], mkpts1[mask]
        out = {'corr1': mkpts0,  'corr2': mkpts1}  # numpy [n1, 2]
        return out


def evaluate_full(model, feat_type='spp', dataset='yfcc', max_length=None, max_keypoints=-1, base_dir=None):
    config_path = 'configs/eval_' + dataset + '_' + feat_type + '.yaml'
    error_th = 1 if dataset == 'yfcc' else 3

    with open(config_path, 'r') as f:
        config = yaml.load(f, yaml.Loader)
        read_config = config['reader']
        read_config['rawdata_dir'] = base_dir / read_config['rawdata_dir']
        read_config['dataset_dir'] = base_dir / read_config['dataset_dir']
        eval_config = config['evaluator']

    if max_keypoints > 0:
        read_config['num_kpt'] = max_keypoints

    reader = StandardReader(config=read_config)
    reader_loader = DataLoader(dataset=reader, num_workers=4, shuffle=False)
    matcher = feed_match_v2
    evaluator = auc_eval(config=eval_config)

    for index in tqdm(range(len(reader_loader)), total=len(reader)):
        if max_length is not None and index >= max_length:
            break
        info = reader.run(index)
        try:
            match_out = matcher(info=info, model=model)
        except Exception as error:
            print(error)
            continue
        cur_res = evaluator.run({**info, **match_out}, th=error_th)
        evaluator.res_inqueue(res=cur_res)
    reader.close()
    output = evaluator.parse()
    aucs = output['exact_auc']
    prec = output['mean_precision']
    mscore = output['mean_match_score']

    # print('Evaluation Results (mean over {} pairs):'.format(len(reader)))
    print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
        aucs[0] * 100, aucs[1] * 100, aucs[3] * 100, prec, mscore))

    result = {
        "auc@5": aucs[0] * 100,
        "auc@10": aucs[1] * 100,
        "auc@15": aucs[2] * 100,
        "auc@20": aucs[3] * 100,
        "prec": prec,
        "mscore": mscore,
    }

    return result
