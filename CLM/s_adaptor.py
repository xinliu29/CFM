import torch
import torch.nn as nn

from KLM.utils import compute_score, compute_matches_without_mutual
from KLM.layers import arange_like
from CLM.cl import CLNet
from CLM.loss import MatchLoss
from CLM.utils import scores_0_to_1, batch_episym


class SAdaptor(nn.Module):

    def __init__(self, conf, layer):
        super().__init__()
        conf["layer_num"] = layer
        self.obj_num_kp = conf.get('obj_num_kp', 0)
        self.opt = conf
        self.model = CLNet(conf)
        self.model_loss = MatchLoss(conf)

    def forward(self, data):
        kpts0, kpts1 = data["keypoints0_3d"], data["keypoints1_3d"]  # [b, m, 2]  # [b, n, 2]
        descs0, descs1 = data["mdesc0"], data["mdesc1"]  # [b, d, m]  [b, d, n]

        dist = torch.einsum('bdn,bdm->bnm', descs0, descs1)
        dist = dist / self.opt['descriptor_dim'] ** .5
        pred_score = compute_score(dist=dist, dustbin=self.opt['bin_score'].to(dist.device),
                                   iteration=self.opt["sinkhorn_iterations"])
        matches0, matches1, matching_scores0, _ = compute_matches_without_mutual(scores=pred_score, p=0)

        descs0 = descs0.transpose(1, 2)  # [b, m, d]
        descs1 = descs1.transpose(1, 2)  # [b, n, d]
        E = data['gt_E']  # [b, 3, 3]

        bsz = E.shape[0]
        desc_dim = descs0.shape[2]
        num = self.obj_num_kp

        xs = torch.zeros(bsz, num, 4, device=kpts0.device)  # [b, num, 4]
        desc_feats = torch.zeros(bsz, num, desc_dim, device=descs0.device)  # [b, num, d]
        indexes = torch.zeros(bsz, num, device=kpts0.device)  # [b, num]
        ret_nearest, ret_pred_score = None, None

        for b in range(bsz):
            kpt0, kpt1 = kpts0[b], kpts1[b]  # [m, 2] [n, 2]
            desc0, desc1 = descs0[b], descs1[b]  # [m, d], [n, d]

            idx_sort = matches0[b]  # [m] if kpt0 has match, store the kpt1 index, else -1

            kpt1 = kpt1[idx_sort, :]  # [m, 2]
            desc1 = desc1[idx_sort, :]  # [m, d]
            x = torch.cat((kpt0, kpt1), dim=-1)  # [m, 4]
            desc_feat = desc0 - desc1  # [m, d]

            if not self.training:
                mutual_nearest = arange_like(matches0, 1)[None] == matches1.gather(1, matches0)
                ret_nearest, ret_pred_score = mutual_nearest, pred_score
                if x.shape[0] < 8:
                    return None
                xs, desc_feats, indexes = x[None], desc_feat[None], idx_sort[None]
                break

            xs[b], desc_feats[b], indexes[b] = x, desc_feat, idx_sort

        ys = batch_episym(xs[:, :, :2], xs[:, :, 2:], E)  # [b, num]
        xs = xs.unsqueeze(1)  # [b, 1, num, 4]
        desc_feats = desc_feats.unsqueeze(1)  # [b, 1, num, d]

        logits = self.model({"xs": xs, "desc_feats": desc_feats, "masks": None})

        confidence0 = torch.relu(torch.tanh(logits[-1])).unsqueeze(1)  # [b, 1, m]
        confidence1 = scores_0_to_1(confidence0, indexes, data["mdesc1"].shape[2])  # [b, 1, n]
        filter_score0 = torch.tanh(logits[-1]).unsqueeze(1).detach()  # [b, 1, m]
        filter_score1 = scores_0_to_1(filter_score0, indexes, data["mdesc1"].shape[2]).detach()  # [b, 1, n]

        ys_ds = [ys for _ in range(len(logits))]

        ret = {"xs": xs, "logits": logits, "ys_ds": ys_ds,
               "confidence": (confidence0, confidence1),
               "filter_score": (filter_score0, filter_score1),
               "mutual_nearest": ret_nearest, "pred_score": ret_pred_score,
               "matching_score": matching_scores0,
               "masks": None}
        return ret

    def loss(self, data):
        pred = self.forward(data)

        logits = pred['logits']  # list [b, n]
        ys_ds = pred['ys_ds']  # list [b, n]
        xs = pred['xs']  # [b, 1, n, 4]
        masks = pred['masks']  # [b, n]

        loss = self.model_loss.run(xs, logits, ys_ds, masks)

        return {"loss": loss, "confidence": pred['confidence'], "pred_score": pred['pred_score']}
