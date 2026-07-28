import torch
import torch.nn.functional as F


class MatchLoss(object):
    default_config = {
        "obj_geod_th": 1e-4,
    }

    def __init__(self, config):
        self.config = {**self.default_config, **config}
        self.obj_geod_th = self.config["obj_geod_th"]

    def weight_estimation(self, gt_geod_d, is_pos, ones):
        dis = torch.abs(gt_geod_d - self.obj_geod_th) / self.obj_geod_th

        weight_p = torch.exp(-dis)
        weight_p = weight_p * is_pos

        weight_n = ones
        weight_n = weight_n * (1 - is_pos)
        weight = weight_p + weight_n

        return weight

    def run(self, xs, logits, ys, masks):
        # Classification loss
        classif_loss = torch.zeros(size=[], device=xs.device)
        with torch.no_grad():
            ones = torch.ones((xs.shape[0], 1)).to(xs.device)
        for i in range(len(logits)):
            gt_geod_d: torch.Tensor = ys[i]  # [b, n]
            is_pos = (gt_geod_d < self.obj_geod_th).to(gt_geod_d.dtype)  # [b, n]
            is_neg = (gt_geod_d >= self.obj_geod_th).to(gt_geod_d.dtype)  # [b, n]
            with torch.no_grad():
                pos = torch.sum(is_pos, dim=-1, keepdim=True)  # [b, 1]
                pos_num = F.relu(pos - 1) + 1
                neg = torch.sum(is_neg, dim=-1, keepdim=True)  # [b, 1]
                neg_num = F.relu(neg - 1) + 1
                pos_w = neg_num / pos_num  # [b, 1]
                pos_w = torch.max(pos_w, ones)  # [b, 1]
                weight = self.weight_estimation(gt_geod_d, is_pos, ones)
            classif_loss += F.binary_cross_entropy_with_logits(input=weight * logits[i],  # [b, n]
                                                               target=is_pos,  # [b, n]
                                                               weight=masks,  # [b, n]
                                                               reduction='mean',
                                                               pos_weight=pos_w)  # [b, 1]

        return classif_loss
