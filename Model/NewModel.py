import math

import numpy as np
import torch
from torch import nn


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

class residual_module(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(residual_module, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=pad,
                              stride=stride)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class DotProductAttention(nn.Module):
    def __init__(self, num_point):
        super(DotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        if num_point == 25:
            self.edges = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
        elif num_point == 17:
            self.edges = torch.tensor(
                [[1, 2], [1, 3], [2, 4], [3, 5], [1, 6], [1, 7], [6, 8], [8, 10], [7, 9], [9, 11], [6, 7], [6, 12], [7, 13],
                 [12, 13],[12, 14], [14, 16], [13, 15]])
        self.adjc_mat = torch.eye(num_point).to('cuda' if torch.cuda.is_available() else 'cpu')
        for i in self.edges:
            self.adjc_mat[i[0]-1][i[1]-1] = 1
            self.adjc_mat[i[1]-1][i[0]-1] = 1

        self.adjc_mat = nn.Parameter(self.adjc_mat, requires_grad=True)

    def forward(self, k, q):
        x = k.unsqueeze(-1) @ q.unsqueeze(-2)
        x += self.adjc_mat
        x = self.softmax(x)
        return x

class spatial_self_attention(nn.Module):
    def __init__(self, in_channels, out_channels, num_point, rel_reduction=8, mid_reduction=1):
        super(spatial_self_attention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.W_K = nn.Conv2d(in_channels, self.rel_channels, kernel_size=1)
        self.W_Q = nn.Conv2d(in_channels, self.rel_channels, kernel_size=1)
        self.W_V = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.attention = DotProductAttention(num_point)
        self.ffn = ffn(self.rel_channels, self.out_channels)
        self.res = residual_module(in_channels, out_channels, 1, 1)
        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x):
        res = self.res(x)
        k = self.W_K(x).mean(-2)
        q = self.W_Q(x).mean(-2)
        v = self.W_V(x)

        x = self.attention(k, q)
        x = self.ffn(x)

        x = torch.matmul(v, x)
        x += res
        x = self.relu(x)

        return x

class fast_path(nn.Module):
    def __init__(self, in_channels, out_channels, time_length):
        super(fast_path, self).__init__()
        in_time_length = time_length
        out_time_length = time_length // 4
        padding = (1, 1)
        stride = (2, 1)
        kernel_size = (3, 3)
        dilation = (1, 1)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class ffn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ffn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x

class ST_feature_extraction(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 time_length,
                 stride=1,
                 residual_kernel_size=1):

        super().__init__()

        self.fast_path = fast_path(in_channels, out_channels, time_length)
        self.slow_path = slow_path(in_channels, out_channels // 2, time_length)
        self.residual = residual_module(in_channels, out_channels, 1, 1)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        out1 = self.fast_path(x)
        out2 = self.slow_path(x)

        out1 = out1.reshape(out1.size(0), -1, out1.size(3)).reshape(x.size(0), -1, x.size(2), x.size(3))
        # out2 = out2.reshape(out2.size(0), -1, out2.size(3)).reshape(x.size(0), -1, x.size(2) // 2, x.size(3))
        out = torch.cat([out1, out2], dim=1)
        out += res
        return out


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_point, time_length):
        super(Block, self).__init__()


        self.spatial_self_attention = spatial_self_attention(in_channels,
                                                             out_channels,
                                                             num_point)

        # self.ST_feature_extraction = ST_feature_extraction(out_channels, out_channels,
        #                                                    time_length)
        self.res = residual_module(in_channels, out_channels, 1, 1)


    def forward(self, x):
        res = self.res(x)
        x = self.spatial_self_attention(x)
        # x = self.ST_feature_extraction(x)
        x += res
        return x

class Model(nn.Module):
    def __init__(self, num_class, num_point, in_channels, num_person=2):
        super(Model, self).__init__()
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channels = 64
        time_length = 64
        self.block1 = Block(in_channels, base_channels, num_point, time_length)
        self.block2 = Block(base_channels, base_channels, num_point, time_length)
        self.block3 = Block(base_channels, base_channels, num_point, time_length)
        self.block4 = Block(base_channels, base_channels, num_point, time_length)
        self.block5 = Block(base_channels, base_channels * 2, num_point, time_length // 2)
        self.block6 = Block(base_channels * 2, base_channels * 2, num_point, time_length // 2)
        self.block7 = Block(base_channels * 2, base_channels * 2, num_point, time_length // 2)
        self.block8 = Block(base_channels * 2, base_channels * 4, num_point, time_length // 4)
        self.block9 = Block(base_channels * 4, base_channels * 4, num_point, time_length // 4)
        self.block10 = Block(base_channels * 4, base_channels * 4, num_point, time_length // 4)


        # self.res = residual_module(base_channels, base_channels, 1, 1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(base_channels * 4, num_class)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        normalized_x = self.data_bn(x)
        x = normalized_x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)

        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.dropout(x)
        x = self.fc(x)
        return self.softmax(x)

