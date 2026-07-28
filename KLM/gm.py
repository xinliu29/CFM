import torch.nn as nn
from KLM.utils import *
from KLM.layers import KeypointEncoder
from KLM.layers import AttentionalGNN
from KLM.loss import GraphLoss


class GM(nn.Module):

    default_config = {
        'descriptor_dim': 256,
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
        'n_layers': 9,
        'n_min_tokens': 256,
        'with_sinkhorn': True,
        'ac_fn': 'relu',
        'norm_fn': 'in',
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}
        self.n_layers = self.config['n_layers']
        self.config['GNN_layers'] = ['self', 'cross'] * self.n_layers
        self.match_threshold = self.config['match_threshold']

        self.with_sinkhorn = self.config['with_sinkhorn']
        self.sinkhorn_iterations = self.config['sinkhorn_iterations']

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'],
            self.config['keypoint_encoder'],
            ac_fn=self.config['ac_fn'],
            norm_fn=self.config['norm_fn'])
        self.gnn = AttentionalGNN(
            feature_dim=self.config['descriptor_dim'],
            layer_names=self.config['GNN_layers'],
            ac_fn=self.config['ac_fn'],
            norm_fn=self.config['norm_fn'],
        )

        self.final_proj = nn.ModuleList([nn.Conv1d(
            self.config['descriptor_dim'],
            self.config['descriptor_dim'],
            kernel_size=1, bias=True) for _ in range(self.n_layers)])

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        self.match_net = GraphLoss(config=self.config)

    def preprocess(self, data):
        desc0, desc1 = data['descriptors0'], data['descriptors1']  # [b, m, d]  [b, n, d]
        kpts0, kpts1 = data['keypoints0_3d'], data['keypoints1_3d']  # [b, m, 2]  [b, n, 2]
        scores0, scores1 = data['scores0'], data['scores1']  # [b, m]  [b, n]
        desc0 = desc0.transpose(1, 2)  # [b, d, m]
        desc1 = desc1.transpose(1, 2)  # [b, d, n]

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int)[0],
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int)[0],
                'matching_scores0': kpts0.new_zeros(shape0)[0],
                'matching_scores1': kpts1.new_zeros(shape1)[0],
                'skip_train': True
            }

        # Keypoint normalization using K
        norm_kpts0 = kpts0
        norm_kpts1 = kpts1

        # Keypoint MLP encoder.
        enc0, enc1 = self.encode_keypoint(norm_kpts0=norm_kpts0, norm_kpts1=norm_kpts1,
                                          scores0=scores0, scores1=scores1)  # [b, d, m]  [b, d, n]

        desc0 = desc0 + enc0  # [b, d, m]
        desc1 = desc1 + enc1  # [b, d, n]

        return desc0, desc1

    def forward_train(self, data):
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

        mdescs = torch.vstack([torch.vstack(mdescs0), torch.vstack(mdescs1)])  #  [n_layer*b*2, d, num(m and n)]

        nI = len(desc0s)  # [n_layer]
        dist = torch.einsum('bdn,bdm->bnm', mdescs[:nI * nB], mdescs[nI * nB:])  # [n_layer*b, m, n]
        dist = dist / self.config['descriptor_dim'] ** .5
        score = compute_score(dist=dist, dustbin=self.bin_score,
                              iteration=self.sinkhorn_iterations,
                              with_sinkhorn=self.with_sinkhorn)  # [n_layer*b, m+1, n+1]

        loss_out = self.match_net(score, data['matching_mask'].repeat(nI, 1, 1))

        all_scores = [score[i * nB: (i + 1) * nB] for i in range(nI)]  # [n_layer, b, m+1, n+1]
        loss_out['scores'] = all_scores
        loss = loss_out['matching_loss']
        loss_out['loss'] = loss
        return loss_out

    def produce_matches(self, data, p=0.2, only_last=False):
        desc0, desc1 = self.preprocess(data)

        nB = desc0.shape[0]

        # Multi-layer Transformer network.
        desc0s, desc1s = self.gnn(desc0, desc1)

        nI = len(desc0s)

        if only_last:
            mdescs0 = self.final_proj[-1](desc0s[-1])
            mdescs1 = self.final_proj[-1](desc1s[-1])
        else:
            mdescs0, mdescs1 = [], []
            for l, d0, d1 in zip(self.final_proj, desc0s, desc1s):
                md0 = l(d0)
                md1 = l(d1)
                mdescs0.append(md0)
                mdescs1.append(md1)

            mdescs0 = torch.vstack(mdescs0)
            mdescs1 = torch.vstack(mdescs1)

        dist = torch.einsum('bdn,bdm->bnm', mdescs0, mdescs1)
        dist = dist / self.config['descriptor_dim'] ** .5
        score = compute_score(dist=dist, dustbin=self.bin_score,
                              iteration=self.sinkhorn_iterations,
                              with_sinkhorn=self.with_sinkhorn)

        indices0, indices1, mscores0, mscores1 = compute_matches(scores=score, p=p)

        if nI == 1 or only_last:
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
        }

        return output

    def forward(self, data):
        if self.training:
            return self.forward_train(data=data)
        else:
            return self.produce_matches(data=data)

    def forward_one_layer(self, desc0, desc1, layer_i, data=None):
        layer = self.gnn.layers[layer_i]
        name = self.gnn.names[layer_i]

        if name == 'cross':
            ds_desc0 = desc0
            ds_desc1 = desc1
            delta0 = layer(desc0, ds_desc1, M=None)
            delta1 = layer(desc1, ds_desc0, M=None)
        elif name == 'self':
            ds_desc0 = desc0
            ds_desc1 = desc1
            delta0 = layer(desc0, ds_desc0, M=None)
            delta1 = layer(desc1, ds_desc1, M=None)
        else:
            raise ValueError("Unknown attention type")

        return desc0 + delta0, desc1 + delta1

    def encode_keypoint(self, norm_kpts0, norm_kpts1, scores0, scores1):
        return self.kenc(norm_kpts0, scores0), self.kenc(norm_kpts1, scores1)

    def compute_distance(self, desc0, desc1, layer_id=-1):
        mdesc0 = self.final_proj[layer_id](desc0)
        mdesc1 = self.final_proj[layer_id](desc1)
        dist = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        dist = dist / self.config['descriptor_dim'] ** .5
        return dist