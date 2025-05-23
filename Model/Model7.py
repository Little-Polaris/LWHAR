import math
import os
import shutil

import torch
from torch import nn



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


class dot_product_attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(dot_product_attention, self).__init__()
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
    def __init__(self, in_channels, out_channels, adjc_mat, rel_reduction=8):
        super(st_attention_block, self).__init__()
        if in_channels < 16:
            rel_channels = 8
        else:
            rel_channels = in_channels // rel_reduction
        self.adjc_mat = nn.Parameter(adjc_mat)
        self.W_K = nn.Conv2d(in_channels, rel_channels//2, kernel_size=1)
        self.W_Q = nn.Conv2d(in_channels, rel_channels//2, kernel_size=1)
        self.W_V = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.dot_production_attention = dot_product_attention(rel_channels//2, rel_channels//2)
        self.ffn = nn.Conv2d(rel_channels, out_channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor([1], dtype=torch.float32))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, adjc_mat):
        k = self.W_K(x).mean(-2)
        q = self.W_Q(x).mean(-2)
        v = self.W_V(x)

        x = self.dot_production_attention(k, q)

        x = self.ffn(x)

        x = self.alpha * x + self.adjc_mat.unsqueeze(0).unsqueeze(0)

        x = x @ v.transpose(-1, -2)
        x = x.transpose(-1, -2).contiguous()
        return x

# class pos_encoding(nn.Module):
#     def __init__(self, channels, num_point):
#         super(pos_encoding, self).__init__()
#
#         pe = torch.zeros(num_point, channels)
#         position = torch.arange(0, num_point, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, channels, 2).float() * (-math.log(10000.0) / channels))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).unsqueeze(0).permute(0, 3, 1, 2)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + self.pe
#         return x

class LearnablePermutation(nn.Module):
    def __init__(self, dim_size, sinkhorn_iters=20):
        super().__init__()
        self.dim_size = dim_size
        self.sinkhorn_iters = sinkhorn_iters
        self.log_alpha = nn.Parameter(torch.randn(dim_size, dim_size))

    def sinkhorn(self, log_alpha):
        for _ in range(self.sinkhorn_iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)  # 行归一化
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)  # 列归一化
        return log_alpha.exp()  # 转换为概率矩阵

    def forward(self, x):
        P = self.sinkhorn(self.log_alpha)
        if self.training:
            x_permuted = torch.einsum('bchw,wp->bchp', x, P)
        else:
            perm = torch.argmax(P, dim=1)
            x_permuted = torch.index_select(x, dim=-1, index=perm)

        return x_permuted

class st_feature_extraction_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, pooling=None, ffn=False):
        super(st_feature_extraction_block, self).__init__()
        self.stride = stride if isinstance(stride, tuple) else (stride, 1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        if pooling == 'max':
            self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=self.stride, padding=(1, 0))
            self.stride = 1
        elif pooling == 'avg':
            self.pooling = nn.AvgPool2d(kernel_size=(3, 1), stride=self.stride, padding=(1, 0))
            self.stride = 1
        else:
            self.pooling = return_x()
        if ffn:
            self.ffn = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=self.stride, padding=padding, dilation=dilation)
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
    def __init__(self , in_channels, out_channels, num_point, stride=1):
        super(sts_feature_extraction, self).__init__()
        self.num_point = num_point
        # self.pe = pos_encoding(in_channels, num_point)
        self.block1 = st_feature_extraction_block(in_channels, out_channels // 4, (5, 1), stride, (2, 0), 1, ffn=True)
        self.block2 = st_feature_extraction_block(in_channels, out_channels // 4, (5, 1), stride, (4, 0), (2, 1), ffn=True)
        self.block3 = st_feature_extraction_block(in_channels, out_channels // 4, 1, stride, 0, 1, 'max')
        self.block4 = st_feature_extraction_block(in_channels, out_channels // 4, 1, stride, 0, 1, 'avg')
        self.apply(weights_init)

    def forward(self, x):
        # x = self.pe(x)
        x_permuted = x
        out1 = self.block1(x_permuted)
        out2 = self.block2(x_permuted)
        out3 = self.block3(x_permuted)
        out4 = self.block4(x_permuted)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        return out



class res_module(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(res_module, self).__init__()
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


class st_attention(nn.Module):
    def __init__(self, in_channels, out_channels, num_point, edges, head_num=3, residual=True):
        super(st_attention, self).__init__()
        self.head_num = head_num
        self.permutations = nn.ModuleList([LearnablePermutation(num_point) for i in range(head_num)])
        self.attention = nn.ModuleList()
        self.adjc_mat = torch.zeros(num_point, num_point)
        for i in edges:
            self.adjc_mat[i[0] - 1][i[1] - 1] = 1
        self.adjc_mat = torch.stack((torch.eye(num_point), self.adjc_mat, self.adjc_mat.T), dim=0)
        for i in range(self.head_num):
            self.attention.append(st_attention_block(in_channels, out_channels, self.adjc_mat[i]))

        if residual:
            if in_channels != out_channels:
                self.res = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.res = return_x()
        else:
            self.res = return_0()

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        out = 0
        for i in range(self.head_num):
            x_permuted = self.permutations[i](x)
            out += self.attention[i](x_permuted, self.adjc_mat[i])
        out = self.bn(out)
        out += self.res(x)
        out = self.relu(out)
        return out


class sts_attention(nn.Module):
    def __init__(self, in_channels, out_channels, num_point, edges, stride=1, residual=True):
        super(sts_attention, self).__init__()
        self.st_attn = st_attention(in_channels, out_channels, num_point, edges)
        self.sts_fe = sts_feature_extraction(out_channels, out_channels, num_point, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = return_0()
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = return_x()
        else:
            self.residual = res_module(in_channels, out_channels, kernel_size=1, stride=(stride, 1))

    def forward(self, x):
        res = self.residual(x)
        x = self.st_attn(x)
        x = self.sts_fe(x)
        y = self.relu(x + res)
        return y


class Model(nn.Module):
    def __init__(self, num_class, num_point, num_person, edges, in_channels,
                 drop_out=0):
        super(Model, self).__init__()
        source_path = os.path.abspath(__file__)
        shutil.copy2(source_path, "./logs/model.py")
        self.num_point = num_point
        # self.adjc_mat = torch.zeros(num_point, num_point)
        # for i in edges:
        #     self.adjc_mat[i[0] - 1][i[1] - 1] = 1
        # self.adjc_mat = torch.stack((torch.eye(num_point), self.adjc_mat, self.adjc_mat.T), dim=0)

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = sts_attention(in_channels, base_channel, num_point, edges, residual=False)
        self.l2 = sts_attention(base_channel, base_channel, num_point, edges)
        self.l3 = sts_attention(base_channel, base_channel * 2, num_point, edges, stride=2)
        self.l4 = sts_attention(base_channel * 2, base_channel * 2, num_point, edges)
        self.l5 = sts_attention(base_channel * 2, base_channel * 4, num_point, edges, stride=2)
        self.l6 = sts_attention(base_channel * 4, base_channel * 4, num_point, edges)

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = return_x()

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        normalized_x = self.data_bn(x)
        x = normalized_x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)

        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
        x = self.fc(x)

        return x