import torch
import torch.nn as nn
from KLM.utils import compute_matches


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class GraphLoss(nn.Module):
    default_config = {
        'match_threshold': 0.2,
        'multi_scale': False,
        'with_pose': False,
        'with_hard_negative': False,
        'neg_margin': 0.1,
        'with_sinkhorn': True,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}
        self.multi_scale = self.config['multi_scale']
        self.with_pose = self.config['with_pose']
        self.with_hard_negative = self.config['with_hard_negative']
        self.neg_margin = self.config['neg_margin']
        self.with_sinkhorn = self.config['with_sinkhorn']

    def forward(self, score, gt_matching_mask):
        output = {}
        match_loss_corr, match_loss_incorr, match_loss_neg \
            = self.compute_matching_loss_batch(pred_scores=score, gt_matching_mask=gt_matching_mask)

        match_loss = match_loss_corr + match_loss_incorr + match_loss_neg
        indices0, indices1, mscores0, mscores1 = compute_matches(scores=score, p=self.config['match_threshold'])

        output['matching_loss'] = match_loss
        # recover matches from the final output
        output['matches0'] = indices0  # use -1 for invalid match
        output['matches1'] = indices1  # use -1 for invalid match
        output['matching_scores0'] = mscores0
        output['matching_scores1'] = mscores1

        return output

    def compute_matching_loss_batch(self, pred_scores, gt_matching_mask):
        log_p = torch.log(abs(pred_scores) + 1e-8)

        num_corr = torch.sum(gt_matching_mask[:, :-1, :-1], dim=2).sum(dim=1)  # [B]
        num_corr[num_corr == 0] = 1
        loss_curr = torch.sum(log_p[:, :-1, :-1] * gt_matching_mask[:, :-1, :-1], dim=2).sum(dim=1)  # [B]
        loss_curr = loss_curr / num_corr
        loss_curr = -loss_curr.mean()

        num_incorr1 = torch.sum(gt_matching_mask[:, :, -1], dim=1)  # [B]
        num_incorr2 = torch.sum(gt_matching_mask[:, -1, :], dim=1)  # [B]
        loss_incorr1 = torch.sum(log_p[:, :, -1] * gt_matching_mask[:, :, -1], dim=1)
        loss_incorr2 = torch.sum(log_p[:, -1, :] * gt_matching_mask[:, -1, :], dim=1)

        incorr1_mask = (num_incorr1 > 0)
        incorr2_mask = (num_incorr2 > 0)

        if torch.sum(incorr1_mask) > 0:
            loss_incorr1 = loss_incorr1[incorr1_mask] / num_incorr1[incorr1_mask]
            loss_incorr2 = loss_incorr2[incorr2_mask] / num_incorr2[incorr2_mask]
            loss_incorr = -(loss_incorr1.mean() + loss_incorr2.mean()) / 2
        else:
            loss_incorr = torch.zeros(size=[], device=gt_matching_mask.device)

        if self.with_hard_negative:
            loss_neg = self.compute_matching_hard_negative_loss(pred_scores=pred_scores,
                                                                gt_matching_mask=gt_matching_mask)
        else:
            loss_neg = torch.zeros(size=[], device=gt_matching_mask.device)

        return loss_curr, loss_incorr, loss_neg

    def compute_matching_hard_negative_loss(self, pred_scores, gt_matching_mask):
        gt_matching_mask_inv = 1 - gt_matching_mask

        pos_row = torch.max(pred_scores[:, :-1, :] * gt_matching_mask[:, :-1, :], dim=2)[
            0]  # discard the last invalid cow
        pos_col = torch.max(pred_scores[:, :, :-1] * gt_matching_mask[:, :, :-1], dim=1)[
            0]  # discard the last invalid col
        neg_row = torch.max(pred_scores[:, :-1, :] * gt_matching_mask_inv[:, :-1, :], dim=2)[0]
        neg_col = torch.max(pred_scores[:, :, :-1] * gt_matching_mask_inv[:, :, :-1], dim=1)[0]

        # mask_row = ((pos_row - neg_row) < self.neg_margin)
        # mask_col = ((pos_col - neg_col) < self.neg_margin)
        loss_neg_row = -torch.clamp_max(pos_row - neg_row - self.neg_margin, max=0).mean()
        loss_neg_col = -torch.clamp_max(pos_col - neg_col - self.neg_margin, max=0).mean()

        loss_neg = (loss_neg_row + loss_neg_col) / 2.

        return loss_neg
