import math
import os
import pdb
import shutil

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.autograd import Variable


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class return_x(nn.Module):
    def __init__(self):
        super(return_x, self).__init__()

    def forward(self, x):
        return x

class return_0(nn.Module):
    def __init__(self):
        super(return_0, self).__init__()

    def forward(self, x):
        return 0

class res_module(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(res_module, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

# class Permutation(nn.Module):
#     def __init__(self, dim_size, tau=1.0, sinkhorn_iters=5, identity_strength=5.0):
#         super(Permutation, self).__init__()
#         self.dim_size = dim_size
#         self.tau = tau
#         self.sinkhorn_iters = sinkhorn_iters
#         self.identity_strength = identity_strength
#         self.log_alpha = nn.Parameter(torch.zeros(dim_size, dim_size))
#         self._init_identity_alpha()

#     def _init_identity_alpha(self):
#         with torch.no_grad():
#             eye = torch.eye(self.dim_size, device=self.log_alpha.device)
#             self.log_alpha.copy_(eye * self.identity_strength + torch.randn_like(self.log_alpha) * 0.01)

#     def sinkhorn(self, log_alpha):
#         for _ in range(self.sinkhorn_iters):
#             log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
#             log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)
#         return log_alpha.exp()

#     def forward(self, x):
#         if self.training:
#             gumbel = -torch.log(-torch.log(torch.rand_like(self.log_alpha) + 1e-20))
#             noisy_alpha = (self.log_alpha + gumbel) / self.tau
#             P = self.sinkhorn(noisy_alpha)
#         else:
#             P = self.sinkhorn(self.log_alpha)
#             P = torch.eye(self.dim_size, device=x.device)[P.argmax(dim=-1)]
#         return torch.einsum('bchw,wp->bchp', x, P)

class attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(attention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
    def forward(self, k, q):
        x1 = k.unsqueeze(-1) @ q.unsqueeze(-2)
        x2 = k.unsqueeze(-1) - q.unsqueeze(-2)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.tanh(x)
        return x

class st_attention_block(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super(st_attention_block, self).__init__()
        if in_channels <= 8:
            rel_channels = 8
        else:
            rel_channels = in_channels // rel_reduction
        self.W_K = nn.Conv2d(in_channels, rel_channels//2, kernel_size=1)
        self.W_Q = nn.Conv2d(in_channels, rel_channels//2, kernel_size=1)
        self.W_V = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.attention = attention(rel_channels//2, rel_channels//2)
        # self.permutation = Permutation(25)
        self.ffn = nn.Conv2d(rel_channels, out_channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.zeros(1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A):
        k = self.W_K(x).mean(-2)
        q = self.W_Q(x).mean(-2)
        v = self.W_V(x)

        x = self.attention(k, q)

        # x = self.permutation(x)

        x = self.ffn(x)

        x = x * self.alpha + A.unsqueeze(0).unsqueeze(0)

        x = x @ v.transpose(-1, -2)
        x = x.transpose(-1, -2).contiguous()

        return x

class st_attention(nn.Module):
    def __init__(self, in_channels, out_channels, num_head=3, residual=True):
        super(st_attention, self).__init__()
        self.num_head = num_head

        self.convs = nn.ModuleList()
        for i in range(self.num_head):
            self.convs.append(st_attention_block(in_channels, out_channels))

        A = np.eye(25)
        adj_matrix = np.zeros((25, 25))
        adj_relation = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                        (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                        (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
        for i in adj_relation:
            adj_matrix[i[0] - 1, i[1] - 1] = 1
        A = np.stack((A, adj_matrix.T, adj_matrix), 0)
        self.A = nn.Parameter(torch.from_numpy(A.astype(np.float32)))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = return_x()
        else:
            self.down = return_0()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        x1 = 0
        res = self.down(x)

        for i in range(self.num_head):
            x1 += self.convs[i](x, self.A[i])

        x = self.bn(x1)
        x += res
        x = self.relu(x)
        return x

class st_feature_extraction_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, pooling=None, ffn=False, mini=False):
        super(st_feature_extraction_block, self).__init__()
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, 1)
        stride = stride if isinstance(stride, tuple) else (stride, 1)
        dilation = dilation if isinstance(dilation, tuple) else (dilation, 1)
        pad0 = (kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1) - 1) // 2
        pad1 = (kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1) - 1) // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1 if not mini else stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if mini:
            self.relu1 = return_x()
            self.pooling = return_x()
            self.ffn = return_x()
            self.bn2 = return_x()
        else:
            self.relu1 = nn.ReLU()
            if pooling == 'max':
                self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=stride, padding=(1, 0))
                self.stride = 1
            elif pooling == 'avg':
                self.pooling = nn.AvgPool2d(kernel_size=(3, 1), stride=stride, padding=(1, 0))
                self.stride = 1
            else:
                self.pooling = return_x()
            if ffn:
                self.ffn = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                     padding=(pad0, pad1), dilation=dilation)
            else:
                self.ffn = return_x()
            self.bn2 = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pooling(x)
        x = self.ffn(x)
        x = self.bn2(x)
        return x

class sts_feature_extraction(nn.Module):
    def __init__(self, num_point, in_channels, out_channels, stride=1, ):
        super(sts_feature_extraction, self).__init__()
        self.block1 = st_feature_extraction_block(in_channels, out_channels // 4, (5, 3), stride, 1, ffn=True)
        self.block2 = st_feature_extraction_block(in_channels, out_channels // 4, (5, 3), stride, (2, 2), ffn=True)
        self.block3 = st_feature_extraction_block(in_channels, out_channels // 4, 1, stride, 1, 'max')
        self.block4 = st_feature_extraction_block(in_channels, out_channels // 4, 1, stride, 1, mini=True)
        self.apply(weights_init)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(x)
        out3 = self.block3(x)
        out4 = self.block4(x)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        return out

class sts_attention(nn.Module):
    def __init__(self, num_point, in_channels, out_channels, stride=1, residual=True):
        super(sts_attention, self).__init__()
        self.st_attention = st_attention(in_channels, out_channels)
        self.sts_fe = sts_feature_extraction(num_point, out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = return_0()

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = return_x()

        else:
            self.residual = res_module(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.sts_fe(self.st_attention(x)) + self.residual(x))
        return y


class Model(nn.Module):
    def __init__(self, num_class, num_point, num_person,in_channels,
                 drop_out=0):
        super(Model, self).__init__()
        source_path = os.path.abspath(__file__)
        shutil.copy2(source_path, "./logs/model.py")
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = sts_attention(num_point, in_channels, base_channel, residual=False)
        self.l2 = sts_attention(num_point, base_channel, base_channel)
        self.l3 = sts_attention(num_point, base_channel, base_channel * 2, stride=2)
        self.l4 = sts_attention(num_point, base_channel * 2, base_channel * 2)
        self.l5 = sts_attention(num_point, base_channel * 2, base_channel * 4, stride=2)
        self.l6 = sts_attention(num_point, base_channel * 4, base_channel * 4)

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        if len(x.shape) == 4:
            x = x.unsqueeze(0)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)