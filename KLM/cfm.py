import torch.nn as nn

from KLM.utils import *
from KLM.gm_cl import GM_CL
from KLM.layers import SAGNN
from KLM.layers import FilterEncoder
from CLM.nn_adaptor import NNAdaptor
from CLM.nn_or_s_eval import s_eval, nn_eval


class CFM(GM_CL):

    def __init__(self, config):
        super().__init__(config)
        self.valid_layer = self.config.get('valid_layers', [5, 8])
        self.use_prune = self.config.get('use_prune', False)
        self.layer_prune = self.config.get('layer_prune', [True, True])
        self.threshold = self.config.get('threshold', [-0.9])
        self.n_min_tokens = self.config.get('n_min_tokens', [0.8])

        self.last_sinkhorn = self.config.get('last_sinkhorn', False)

        self.filters = nn.ModuleList(
            [NNAdaptor(self.config["filter"], self.valid_layer[i]) for i in range(len(self.valid_layer))]
        )
        if self.n_layers - 1 in self.valid_layer:
            fenc_num = len(self.valid_layer) - 1
        else:
            fenc_num = len(self.valid_layer)

        self.fenc = nn.ModuleList(
            [FilterEncoder(self.config['descriptor_dim'],
                           self.config['keypoint_encoder'],
                           ac_fn=self.config['ac_fn'],
                           norm_fn=self.config['norm_fn'])
             for _ in range(fenc_num)])

        self.gnn = SAGNN(
            feature_dim=self.config['descriptor_dim'],
            layer_names=self.config['GNN_layers'],
            ac_fn=self.config['ac_fn'],
            norm_fn=self.config['norm_fn'],
            sharing_layers=None,
        )

        self.M00, self.M01, self.M10, self.M11 = None, None, None, None
        self.self_prob0, self.self_prob1, self.cross_prob0, self.cross_prob1 = None, None, None, None
        self.global_ids0, self.global_ids1 = None, None
        self.all_gids0, self.all_gids1 = None, None

    def update(self, confidence0: torch.Tensor, confidence1: torch.Tensor, gids0, gids1, bi, idx):
        m, n = confidence0.shape[-1], confidence1.shape[-1]
        with torch.no_grad():
            min_num_0 = int(m * self.n_min_tokens[idx])
            min_num_1 = int(n * self.n_min_tokens[idx])
            update_0 = False if gids0.shape[-1] <= min_num_0 else True
            update_1 = False if gids1.shape[-1] <= min_num_1 else True
            if update_0:
                conf0: torch.Tensor = confidence0[bi].squeeze()[gids0]
                full_ids0 = conf0 > self.threshold[idx]  # new_m
                if full_ids0.sum() < min_num_0:
                    full_ids0 = torch.zeros_like(conf0, dtype=torch.bool)
                    _, top_k_indices = torch.topk(conf0, min_num_0)
                    full_ids0[top_k_indices] = True
                gids0 = gids0[full_ids0]

            if update_1:
                conf1: torch.Tensor = confidence1[bi].squeeze()[gids1]
                full_ids1 = conf1 > self.threshold[idx]  # new_n
                if full_ids1.sum() < min_num_1:
                    full_ids1 = torch.zeros_like(conf1, dtype=torch.bool)
                    _, top_k_indices = torch.topk(conf1, min_num_1)
                    full_ids1[top_k_indices] = True
                gids1 = gids1[full_ids1]

            if self.training:
                self.all_gids0[bi] = gids0
                self.all_gids1[bi] = gids1
                self.M00[bi, :, gids0.long()] = 1  # [b, m, m]
                self.M10[bi, :, gids0.long()] = 1  # [b, n, m]
                self.M11[bi, :, gids1.long()] = 1  # [b ,n, n]
                self.M01[bi, :, gids1.long()] = 1  # [b, m, n]

            else:
                self.global_ids0 = gids0
                self.global_ids1 = gids1

    def preprocess(self, data):
        desc0, desc1 = data['descriptors0'], data['descriptors1']  # [b, m, d]  [b, n, d]
        norm_kpts0, norm_kpts1 = data['keypoints0_3d'], data['keypoints1_3d']  # [b, m, 2]  [b, n, 2]
        scores0, scores1 = data['scores0'], data['scores1']  # [b, m]  [b, n]
        desc0 = desc0.transpose(1, 2)  # [b, d, m]
        desc1 = desc1.transpose(1, 2)  # [b, d, n]

        # Keypoint MLP encoder.
        enc0, enc1 = self.encode_keypoint(norm_kpts0=norm_kpts0, norm_kpts1=norm_kpts1,
                                          scores0=scores0, scores1=scores1)  # [b, d, m]  [b, d, n]

        desc0 = desc0 + enc0  # [b, d, m]
        desc1 = desc1 + enc1  # [b, d, n]

        self.M00, self.M01, self.M10, self.M11 = None, None, None, None
        nB, m, n = desc0.shape[0], desc0.shape[-1], desc1.shape[-1]

        self.global_ids0 = torch.arange(0, m, device=desc0.device, requires_grad=False)  # [m]
        self.global_ids1 = torch.arange(0, n, device=desc0.device, requires_grad=False)  # [n]
        self.all_gids0 = [self.global_ids0 for _ in range(nB)]  # list [b, m]
        self.all_gids1 = [self.global_ids1 for _ in range(nB)]  # list [b, n]

        return desc0, desc1, norm_kpts0, norm_kpts1, scores0, scores1

    def forward_train(self, data):
        matching_loss, filter_loss, matching_scores0 = 0, 0, []

        desc0, desc1, norm_kpts0, norm_kpts1, scores0, scores1 = self.preprocess(data)
        nI, nB, m, n = self.n_layers, desc0.shape[0], desc0.shape[-1], desc1.shape[-1]

        dust_id0 = torch.Tensor([m]).to(desc0.device).to(torch.int64)  # [1]: a value m
        dust_id1 = torch.Tensor([n]).to(desc1.device).to(torch.int64)  # [1]: a value n

        # Multi-layer Transformer network.
        for i, (layer, name) in enumerate(zip(self.gnn.layers, self.gnn.names)):
            idx = int(i / 2)
            if name == 'cross':
                src0, src1 = desc1, desc0
                delta0 = layer(desc0, src0, M=self.M01)
                delta1 = layer(desc1, src1, M=self.M10)
            else:
                src0, src1 = desc0, desc1
                delta0 = layer(desc0, src0, M=self.M00)
                delta1 = layer(desc1, src1, M=self.M11)

            desc0 = desc0 + delta0  # [b, d, m]
            desc1 = desc1 + delta1  # [b, d, n]

            if name == 'cross':
                mdesc0 = self.final_proj[idx](desc0)  # [b, d, m]
                mdesc1 = self.final_proj[idx](desc1)  # [b, d, n]
                to_augment = idx in self.valid_layer and idx != self.n_layers - 1

                if idx in self.valid_layer:
                    idx_ = self.valid_layer.index(idx)
                    ret = self.filters[idx_].loss({"mdesc0": mdesc0, "mdesc1": mdesc1, **data})
                    filter_loss += ret["loss"]

                    # use confidence score to augment the descriptors
                    if to_augment:
                        confidence0, confidence1 = ret["confidence"]  # [b, 1, m], [b, 1, n]
                        filter_score0, filter_score1 = ret["filter_score"]  # [b, 1, m], [b, 1, n]
                        fenc0, fenc1 = self.encode_clnet(norm_kpts0=norm_kpts0, norm_kpts1=norm_kpts1,
                                                         scores0=scores0, scores1=scores1,
                                                         confidence0=confidence0, confidence1=confidence1,
                                                         layer=idx_)  # [b, d, m] [b, d, n]

                        desc0 = desc0 + fenc0  # [b, d, m]
                        desc1 = desc1 + fenc1  # [b, d, n]

                if idx < self.valid_layer[0] or not self.use_prune:
                    dist = torch.einsum('bdm,bdn->bmn', mdesc0, mdesc1)  # [b, m, n]
                    dist = dist / self.config['descriptor_dim'] ** .5  # [b, m, n]
                    score = compute_score(dist=dist, dustbin=self.bin_score,
                                          iteration=self.sinkhorn_iterations,
                                          with_sinkhorn=self.with_sinkhorn)  # [b, m+1, n+1]
                    loss_out = self.match_net(score, data['matching_mask'])

                    matching_loss += loss_out['matching_loss']
                    matching_scores0.append(loss_out['matching_scores0'])  # [b, m]
                else:
                    batch_loss = torch.zeros(size=[], device=desc0.device)
                    batch_mscores0 = torch.zeros((nB, m), device=desc0.device, requires_grad=False)
                    self.M00 = torch.zeros((nB, m, m), device=desc0.device, requires_grad=False)
                    self.M10 = torch.zeros((nB, n, m), device=desc0.device, requires_grad=False)
                    self.M11 = torch.zeros((nB, n, n), device=desc0.device, requires_grad=False)
                    self.M01 = torch.zeros((nB, m, n), device=desc0.device, requires_grad=False)

                    for bi in range(nB):
                        gids0 = self.all_gids0[bi]  # [new_m]
                        gids1 = self.all_gids1[bi]  # [new_n]
                        sel_mdesc0 = mdesc0[bi, :, gids0][None]  # [1, d, new_m]
                        sel_mdesc1 = mdesc1[bi, :, gids1][None]  # [1, d, new_n]

                        dist = torch.einsum('bdm,bdn->bmn', sel_mdesc0, sel_mdesc1)  # [b, new_m, new_n]
                        dist = dist / self.config['descriptor_dim'] ** .5  # [b, new_m, new_n]
                        score = compute_score(dist=dist, dustbin=self.bin_score,
                                              iteration=self.sinkhorn_iterations,
                                              with_sinkhorn=self.with_sinkhorn)  # [b, new_m+1, new_n+1]

                        new_m, new_n = sel_mdesc0.shape[-1], sel_mdesc1.shape[-1]
                        # [(new_m+1) x (new_n+1), 1]
                        index0 = torch.hstack([gids0, dust_id0])[:, None].repeat(1, new_n + 1).reshape(-1, 1)
                        # [(new_m+1) x (new_n+1), 1]
                        index1 = torch.hstack([gids1, dust_id1])[None, :].repeat(new_m + 1, 1).reshape(-1, 1)
                        # [1, new_m+1, new_n+1]
                        gt_score = data['matching_mask'][bi][index0, index1].reshape(new_m + 1, new_n + 1)[None]
                        # avoid some matched keypoints do not exist
                        gt_score[:, :-1, new_n] = 1 - torch.max(gt_score[:, :-1, :-1], dim=2)[0]  # last column
                        gt_score[:, new_m, :-1] = 1 - torch.max(gt_score[:, :-1, :-1], dim=1)[0]  # last row

                        loss_out = self.match_net(score, gt_score)
                        if not torch.isnan(loss_out['matching_loss']):
                            batch_loss += loss_out['matching_loss']
                            batch_mscores0[bi, gids0] = loss_out['matching_scores0']

                        if to_augment:
                            self.update(filter_score0, filter_score1, gids0, gids1, bi, idx_)

                    matching_loss += batch_loss / nB
                    matching_scores0.append(batch_mscores0)

        return {
            'matching_loss': matching_loss / nI,
            'filter_loss': filter_loss / nI,
            'loss': (matching_loss + filter_loss) / nI,
            'matching_scores0': matching_scores0,
        }

    def produce_matches(self, data, **kwargs):
        filter_ret = []
        desc0, desc1, norm_kpts0, norm_kpts1, scores0, scores1 = self.preprocess(data)

        # Multi-layer Transformer network.
        for i, (layer, name) in enumerate(zip(self.gnn.layers, self.gnn.names)):
            idx = int(i / 2)
            if name == 'cross':
                src0, src1 = desc1, desc0
                delta0 = layer(desc0, src0)
                delta1 = layer(desc1, src1)
            else:
                src0, src1 = desc0, desc1
                delta0 = layer(desc0, src0)
                delta1 = layer(desc1, src1)

            desc0 = desc0 + delta0  # [b, d, m]
            desc1 = desc1 + delta1  # [b, d, n]

            if name == 'cross':
                mdesc0 = self.final_proj[idx](desc0)  # [b, d, m]
                mdesc1 = self.final_proj[idx](desc1)  # [b, d, n]
                if idx in self.valid_layer:
                    idx_ = self.valid_layer.index(idx)
                    if idx == self.n_layers - 1:
                        if self.last_sinkhorn:
                            self.config['filter']['bin_score'] = self.bin_score
                            self.config['filter']['sinkhorn_iterations'] = self.config['sinkhorn_iterations']
                            ret = s_eval({"mdesc0": mdesc0, "mdesc1": mdesc1, **data}, self.config["filter"])
                            filter_ret.append(ret)
                        else:
                            ret = self.filters[idx_]({"mdesc0": mdesc0, "mdesc1": mdesc1, **data})
                            filter_ret.append(ret)
                        break

                    ret = self.filters[idx_]({"mdesc0": mdesc0, "mdesc1": mdesc1, **data})
                    filter_ret.append(ret)

                    # use confidence score to augment the descriptors
                    confidence0, confidence1 = ret["confidence"]  # [b, 1, m], [b, 1, n]
                    filter_score0, filter_score1 = ret["filter_score"]  # [b, 1, m], [b, 1, n]

                    fenc0, fenc1 = self.encode_clnet(norm_kpts0=norm_kpts0, norm_kpts1=norm_kpts1,
                                                     scores0=scores0, scores1=scores1,
                                                     confidence0=confidence0, confidence1=confidence1,
                                                     layer=idx_)  # [b, d, m] [b, d, n]

                    desc0 = desc0 + fenc0  # [b, d, m]
                    desc1 = desc1 + fenc1  # [b, d, n]

                    if self.use_prune:
                        self.update(filter_score0, filter_score1, self.global_ids0, self.global_ids1, 0, idx_)

                        data["keypoints0_3d"] = data["keypoints0_3d"][:, self.global_ids0, :]
                        data["keypoints1_3d"] = data["keypoints1_3d"][:, self.global_ids1, :]
                        desc0 = desc0[..., self.global_ids0]
                        desc1 = desc1[..., self.global_ids1]
                        norm_kpts0 = norm_kpts0[:, self.global_ids0, :]
                        norm_kpts1 = norm_kpts1[:, self.global_ids1, :]
                        scores0 = scores0[..., self.global_ids0]
                        scores1 = scores1[..., self.global_ids1]

                        m, n = desc0.shape[-1], desc1.shape[-1]

                        self.global_ids0 = torch.arange(0, m, device=desc0.device, requires_grad=False)
                        self.global_ids1 = torch.arange(0, n, device=desc0.device, requires_grad=False)

                if idx == self.n_layers - 1:
                    if self.last_sinkhorn:
                        self.config['filter']['bin_score'] = self.bin_score
                        self.config['filter']['sinkhorn_iterations'] = self.config['sinkhorn_iterations']
                        ret = s_eval({"mdesc0": mdesc0, "mdesc1": mdesc1, **data}, self.config["filter"])
                    else:
                        ret = nn_eval({"mdesc0": mdesc0, "mdesc1": mdesc1, **data})

                    filter_ret.append(ret)

        return {'filter_ret': filter_ret}

    def encode_clnet(self, norm_kpts0, norm_kpts1, scores0, scores1, confidence0, confidence1, layer):
        fenc0 = self.fenc[layer](norm_kpts0, scores0, confidence0)
        fenc1 = self.fenc[layer](norm_kpts1, scores1, confidence1)
        return fenc0, fenc1
