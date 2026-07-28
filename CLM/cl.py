import torch
import torch.nn as nn
import torch.nn.functional as F


class PointCN(nn.Module):
    def __init__(self, channels, out_channels=None, use_bn=True, use_short_cut=True):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels

        self.use_short_cut = use_short_cut
        if use_short_cut:
            self.shot_cut = None
            if out_channels != channels:
                self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        if use_bn:
            self.conv = nn.Sequential(
                    nn.InstanceNorm2d(channels, eps=1e-3),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(True),
                    nn.Conv2d(channels, out_channels, kernel_size=1),
                    nn.InstanceNorm2d(out_channels, eps=1e-3),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=1)
                    )
        else:
            self.conv = nn.Sequential(
                    nn.InstanceNorm2d(channels, eps=1e-3),
                    nn.ReLU(),
                    nn.Conv2d(channels, out_channels, kernel_size=1),
                    nn.InstanceNorm2d(out_channels, eps=1e-3),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=1)
                    )

    def forward(self, x):
        out = self.conv(x)
        if self.use_short_cut:
            if self.shot_cut:
                out = out + self.shot_cut(x)
            else:
                out = out + x
        return out


def knn(x, space_distance, k, masks=None):
    # x: [b, 128, N]
    # space_distance: [b, N, N]  0 ~ 1 [higher value, more relevent]
    # masks: [b, n]
    batch_size, _, num = x.shape
    x = F.normalize(x, p=2, dim=1)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # [b, n, n]
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # [b, 1, n]
    feature_distance = - (xx + inner + xx.transpose(2, 1))  # [b, n, n] negative distance: [higher value, more relevent]
    max_dist, min_dist = feature_distance.max(), feature_distance.min()
    feature_distance = (feature_distance - min_dist) / (max_dist - min_dist)  # [b, n, n]

    if space_distance is not None:
        pairwise_distance = feature_distance * space_distance  # [b, n, n]
    else:
        pairwise_distance = feature_distance

    if masks is not None:
        temp_masks = masks[:, None, :].repeat(1, num, 1)  # [b, n, n]
        pairwise_distance[temp_masks == 0] = - torch.inf

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # [b, n, k]

    return idx[:, :, :]


def get_graph_feature(x, space_consist, masks=None, k=20, idx=None):
    """
    x:     [b, 128, n, 1]
    masks: [b, n]
    """
    batch_size, num_dims, num_points, _ = x.size()
    x = x.view(batch_size, -1, num_points)  # [b, 128, N]
    idx_out = knn(x, space_consist, k=k, masks=masks) if idx is None else idx  # [b, N, k]
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points  # [b, 1, 1]
    idx = idx_out + idx_base  # [b, N, k]
    idx = idx.view(-1)  # [b * N * k]

    x = x.transpose(2, 1).contiguous()  # [b, N, 128]
    feature = x.view(batch_size * num_points, -1)[idx, :]  # [b * N * k, 128]
    feature = feature.view(batch_size, num_points, k, num_dims)  # [b, N, k, 128]
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # [b, N, k, 128]
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()  # [b, 256, N, k]

    if masks is not None:
        temp_masks = masks[:, None, :, None].repeat(1, num_dims * 2, 1, k)  # [b, 256, N, k]
        feature[temp_masks == 0] = 0

    return feature


class ResNet_Block1(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ResNet_Block1, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel, eps=1e-3),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel, eps=1e-3),
            nn.BatchNorm2d(outchannel),
        )

    def forward(self, x):
        x1 = x
        out = self.left(x)
        out = out + x1
        return torch.relu(out)


class ResNet_Block2(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ResNet_Block2, self).__init__()
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel, eps=1e-3),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel, eps=1e-3),
            nn.BatchNorm2d(outchannel),
        )

    def forward(self, x):
        x1 = self.right(x)
        out = self.left(x)
        out = out + x1
        return torch.relu(out)


class DGCNN_Block(nn.Module):
    def __init__(self, knn_num=9, in_channel=128):
        super(DGCNN_Block, self).__init__()
        self.knn_num = knn_num
        self.in_channel = in_channel

        assert self.knn_num == 9 or self.knn_num == 6
        if self.knn_num == 9:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel * 2, self.in_channel, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 3)),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            )
        if self.knn_num == 6:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel * 2, self.in_channel, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 2)),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            )

    def forward(self, features, space_consist, masks):
        B, _, N, _ = features.shape  # [b, 128, N, 1]
        out = get_graph_feature(features, space_consist, masks=masks, k=self.knn_num)  # [b, 256, N, k]
        out = self.conv(out)  # [b, 128, N, 1]
        return out


class GCN_Block(nn.Module):
    def __init__(self, in_channel):
        super(GCN_Block, self).__init__()
        self.in_channel = in_channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )

    def attention(self, w):
        w = torch.relu(torch.tanh(w)).unsqueeze(-1)
        A = torch.bmm(w.transpose(1, 2), w)
        return A

    def graph_aggregation(self, x, w):
        B, _, N, _ = x.size()
        with torch.no_grad():
            A = self.attention(w)
            I = torch.eye(N).unsqueeze(0).to(x.device).detach()
            A = A + I
            D_out = torch.sum(A, dim=-1)
            D = (1 / D_out) ** 0.5
            D = torch.diag_embed(D)
            L = torch.bmm(D, A)
            L = torch.bmm(L, D)
        out = x.squeeze(-1).transpose(1, 2).contiguous()
        out = torch.bmm(L, out).unsqueeze(-1)
        out = out.transpose(1, 2).contiguous()

        return out

    def forward(self, x, w):
        out = self.graph_aggregation(x, w)
        out = self.conv(out)
        return out


class DS_Block(nn.Module):

    def __init__(self, net_channels, input_channel, descriptor_dim, use_desc, use_global):
        nn.Module.__init__(self)
        self.channels = net_channels
        self.use_desc = use_desc
        self.use_global = use_global
        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, self.channels, kernel_size=(1, 1)),
            PointCN(self.channels, use_bn=True)
        ) if self.use_desc else nn.Sequential(
            nn.Conv2d(input_channel, self.channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True)
        )

        self.desc_conv = nn.Sequential(
            nn.Conv2d(descriptor_dim, self.channels, kernel_size=(1, 1)),
            PointCN(self.channels, use_bn=True)
        ) if self.use_desc else None

        if self.use_global:
            self.gcn = GCN_Block(self.channels)

        self.embed_before = nn.Sequential(
            ResNet_Block1(self.channels, self.channels),
            ResNet_Block1(self.channels, self.channels),
            ResNet_Block1(self.channels, self.channels),
            ResNet_Block1(self.channels, self.channels),
        )
        self.dgcnn = DGCNN_Block(9, self.channels)
        self.embed_after = nn.Sequential(
            ResNet_Block1(self.channels, self.channels),
            ResNet_Block1(self.channels, self.channels),
            ResNet_Block1(self.channels, self.channels),
            ResNet_Block1(self.channels, self.channels),
        )

        self.embed_1 = ResNet_Block2(self.channels, self.channels)  # modify the output channel * 2 if needed
        self.linear_0 = nn.Conv2d(self.channels, 1, kernel_size=(1, 1))
        self.linear_1 = nn.Conv2d(self.channels, 1, kernel_size=(1, 1))  # modify the output channel * 2 if needed
        # for param in self.linear_0.parameters():
        #     param.requires_grad = False

    def forward(self, data, desc, space_consist, masks):
        # data:  [b, 4 or 5, n, 1]
        # desc:  [b, d,      n, 1]
        # masks: [b, n]
        b, dim, n, _ = data.shape

        out = self.conv(data)

        if self.use_desc:
            desc_out = self.desc_conv(desc)
            out = out + desc_out

        out = self.embed_before(out)  # [b, 128,  n, 1]

        out = self.dgcnn(out, space_consist, masks)  # [b, 128, n, 1]

        out = self.embed_after(out)  # [b, 128,  n, 1]

        w0 = self.linear_0(out).view(b, -1)  # [b, n]

        if self.use_global:
            out_g = self.gcn(out, w0.detach())  # [b, 128,  n, 1]
            out = out_g + out  # [b, 128,  n, 1]

        out = self.embed_1(out)  # [b, 128,  n, 1]

        logits = self.linear_1(out).view(b, -1)  # [b, n]

        return [w0, logits]


class CLNet(nn.Module):
    default_config = {
        'iter_num': 1,
        'net_channels': 128,
        'use_desc': True,
        'use_global': True,
    }

    def __init__(self, config):
        config = {**self.default_config, **config}
        nn.Module.__init__(self)

        self.iter_num = config['iter_num']
        self.weights_init = DS_Block(config['net_channels'], 4, config['descriptor_dim'], config['use_desc'], config['use_global'])
        self.weights_iter = [DS_Block(config['net_channels'], 5, config['descriptor_dim'], config['use_desc'], config['use_global'])
                             for _ in range(self.iter_num)]
        self.weights_iter = nn.Sequential(*self.weights_iter)
        self.sigma_spat = config['sigma_spat']

    def forward(self, data):
        """
        data['xs']: [b, 1, n, 4]
        data['desc_feats']: [b, 1, n, d]
        data['masks']: [b, n]
        """
        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1  # [b, 1, n, 4]
        xs = data['xs'].transpose(1, 3)  # [b, 4, n, 1]
        desc_feats = data['desc_feats'].transpose(1, 3) if "desc_feats" in data.keys() else None  # [b, d, n, 1]
        masks = data['masks']  # [b, n]

        with torch.no_grad():
            src_keypts = data['xs'][..., :2].squeeze(1)  # src_keypts: [b, n, 2]
            tgt_keypts = data['xs'][..., 2:].squeeze(1)  # tgt_keypts: [b, n, 2]
            src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)  # [b, n, n]
            tgt_dist = torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1)  # [b, n, n]
            space_consist = src_dist - tgt_dist  # [b, n, n]
            space_consist = torch.clamp(1.0 - space_consist ** 2 / self.sigma_spat ** 2, min=0)  # [b, n, n]

        res_logits = []

        logits = self.weights_init(xs,
                                   desc_feats, space_consist, masks)  # list[2 * [b, n]]
        res_logits += logits

        for i in range(self.iter_num):
            w = torch.relu(torch.tanh(logits[-1])).detach()  # [b, n]
            w = w[:, None, :, None]  # [b, 1, n, 1]
            logits = self.weights_iter[i](torch.cat([xs, w], dim=1),
                                          desc_feats, space_consist, masks)  # list[2 * [b, n]]
            res_logits += logits

        return res_logits  # 2 * (self.iter_num + 1)
