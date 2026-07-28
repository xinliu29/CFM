import torch.nn as nn

from KLM.utils import *
from KLM.gm import GM
from CLM.nn_adaptor import NNAdaptor
from CLM.s_adaptor import SAdaptor


class GM_CL(GM):

    def __init__(self, config):
        super().__init__(config)
        filter_type = self.config['filter_type']
        self.use_nn = filter_type == "nn"
        self.config['filter']['descriptor_dim'] = self.config['descriptor_dim']

        if filter_type == "nn":
            self.filters = nn.ModuleList(
                [NNAdaptor(self.config["filter"], i + 1) for i in range(self.n_layers)]
            )
        elif filter_type == "sinkhorn":
            self.config['filter']['bin_score'] = self.bin_score
            self.config['filter']['sinkhorn_iterations'] = self.config['sinkhorn_iterations']
            self.filters = nn.ModuleList(
                [SAdaptor(self.config["filter"], i + 1) for i in range(self.n_layers)]
            )
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

    def forward_train(self, data):
        cl_loss = 0
        pred_score = []
        desc0, desc1 = self.preprocess(data)

        nB = desc0.shape[0]  # [b]

        # Multi-layer Transformer network.
        desc0s, desc1s = self.gnn(desc0, desc1)

        mdescs0, mdescs1 = [], []
        for l, d0, d1 in zip(self.final_proj, desc0s, desc1s):
            # d0: [b, d, m], d1: [b, d, n]
            md = l(torch.vstack([d0, d1]))  # [2*b, d, num(m and n)]
            mdescs0.append(md[:nB])
            mdescs1.append(md[nB:])

        mdescs = torch.vstack([torch.vstack(mdescs0), torch.vstack(mdescs1)])  # [n_layer*b*2, d, num(m and n)]

        nI = len(desc0s)  # [n_layer]
        for i in range(nI):
            mdesc0 = mdescs0[i]
            mdesc1 = mdescs1[i]
            ret = self.filters[i].loss({**{"mdesc0": mdesc0, "mdesc1": mdesc1}, **data})
            cl_loss += ret["loss"] / nI
            if not self.use_nn:
                pred_score.append(ret["pred_score"])

        if self.use_nn:
            dist = torch.einsum('bdn,bdm->bnm', mdescs[:nI * nB], mdescs[nI * nB:])  # [n_layer*b, m, n]
            dist = dist / self.config['descriptor_dim'] ** .5
            score = compute_score(dist=dist, dustbin=self.bin_score,
                                  iteration=self.sinkhorn_iterations,
                                  with_sinkhorn=self.with_sinkhorn)  # [n_layer*b, m+1, n+1]
        else:
            score = pred_score

        loss_out = self.match_net(score, data['matching_mask'].repeat(nI, 1, 1))

        all_scores = [score[i * nB: (i + 1) * nB] for i in range(nI)]  # [n_layer, b, m+1, n+1]
        loss_out['scores'] = all_scores
        loss_out['filter_loss'] = cl_loss
        loss_out['loss'] = loss_out['matching_loss'] + loss_out['filter_loss']
        return loss_out

    def produce_matches(self, data, p=0.2, only_last=False):
        filter_ret = []
        pred_score = []
        desc0, desc1 = self.preprocess(data)

        nB = desc0.shape[0]  # [b]

        # Multi-layer Transformer network.
        desc0s, desc1s = self.gnn(desc0, desc1)

        nI = len(desc0s)  # [n_layer]

        if only_last:
            mdescs0 = self.final_proj[-1](desc0s[-1])
            mdescs1 = self.final_proj[-1](desc1s[-1])
            ret = self.filters[-1]({**{"mdesc0": mdescs0, "mdesc1": mdescs1}, **data})
            filter_ret.append(ret)
            if not self.use_nn:
                pred_score.append(ret["pred_score"])
        else:
            mdescs0, mdescs1 = [], []
            for l, d0, d1 in zip(self.final_proj, desc0s, desc1s):
                md0, md1 = l(d0), l(d1)
                mdescs0.append(md0)
                mdescs1.append(md1)
            mdescs0 = torch.vstack(mdescs0)
            mdescs1 = torch.vstack(mdescs1)
            for i in range(nI):
                mdesc0 = mdescs0[i].unsqueeze(0)
                mdesc1 = mdescs1[i].unsqueeze(0)
                ret = self.filters[i]({**{"mdesc0": mdesc0, "mdesc1": mdesc1}, **data})
                filter_ret.append(ret)
                if not self.use_nn:
                    pred_score.append(ret["pred_score"])

        if not self.use_nn:
            dist = torch.einsum('bdn,bdm->bnm', mdescs0, mdescs1)  # [n_layer*b, m, n]
            dist = dist / self.config['descriptor_dim'] ** .5
            score = compute_score(dist=dist, dustbin=self.bin_score,
                                  iteration=self.sinkhorn_iterations,
                                  with_sinkhorn=self.with_sinkhorn)
        else:
            if only_last:
                score = pred_score[-1]
            else:
                score = pred_score

        indices0, indices1, mscores0, mscores1 = compute_matches(scores=score, p=p)
        if only_last:
            all_scores = [score]
            all_indices0 = [indices0]
            all_mscores0 = [mscores0]
        else:
            all_scores = [score[i * nB: (i + 1) * nB] for i in range(nI)]
            all_indices0 = [indices0[i * nB: (i + 1) * nB] for i in range(nI)]
            all_mscores0 = [mscores0[i * nB: (i + 1) * nB] for i in range(nI)]

        output = {
            'scores': all_scores,
            'indices0': all_indices0,
            'mscores0': all_mscores0,
            'filter_ret': filter_ret
        }
        return output
