import math
import os
import pdb
import shutil

import numpy as np
import torch
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


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class LearnablePermutation(nn.Module):
    def __init__(self, dim_size, sinkhorn_iters=20):
        super().__init__()
        self.dim_size = dim_size
        self.sinkhorn_iters = sinkhorn_iters

        # 可学习参数矩阵（方阵）
        self.log_alpha = nn.Parameter(torch.randn(dim_size, dim_size))

    def sinkhorn(self, log_alpha):
        """Sinkhorn算法生成双随机矩阵"""
        for _ in range(self.sinkhorn_iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)  # 行归一化
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)  # 列归一化
        return log_alpha.exp()  # 转换为概率矩阵

    def forward(self, x, A):
        """
        Args:
            x: 输入张量，形状为 (B, C, H, W)，需对第四维(W)排列
        Returns:
            重排列后的张量，形状不变
        """
        # 生成双随机矩阵
        P = self.sinkhorn(self.log_alpha)

        # 获取排列索引（训练时使用soft排列，推理时使用hard排列）
        if self.training:
            # 训练时：使用可微的soft排列（矩阵乘法）
            x_permuted = torch.einsum('bchw,wp->bchp', x, P)
            A_permuted = torch.einsum('bchw,wp->bchp', A, P)
        else:
            # 推理时：使用精确的hard排列（argmax）
            perm = torch.argmax(P, dim=1)
            x_permuted = torch.index_select(x, dim=-1, index=perm)
            A_permuted = torch.index_select(A, dim=-1, index=perm)

        return x_permuted, A

# class Permutation(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.permutation_matrix = nn.Parameter(torch.arange(dim, dtype=torch.float32) / 100)
#
#     def forward(self, x, A):
#         new_index = torch.argsort(self.permutation_matrix)
#         x_permuted = torch.gather(x, dim=3, index=new_index.expand(*x.shape[:3], -1))
#         A_permuted = torch.gather(A, dim=3, index=new_index.expand(*A.shape[:3], -1))
#         x_permuted += self.permutation_matrix
#         # x_permuted = torch.zeros_like(x)
#         # A_permuted = torch.zeros_like(A)
#         # for i in range(len(new_index)):
#         #     x_permuted[:, :, :, i] = x[:, :, :, new_index[i]]
#         #     A_permuted[:, :, :, i] = A[:, :, :, new_index[i]]
#         return x_permuted, A_permuted

class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels//2, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels//2, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        # self.permutation = LearnablePermutation(25)
        # self.permutation = Permutation(25)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        # self.conv5 = nn.Conv2d(self.rel_channels//2, self.rel_channels//2, kernel_size=1)
        # self.conv6 = nn.Conv2d(self.rel_channels//2, self.rel_channels//2, kernel_size=1)
        self.tanh = nn.Tanh()
        # self.alpha = nn.Parameter(torch.Tensor([0.5]))
        # self.beta = nn.Parameter(torch.Tensor([0.5]))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x4 = x1.unsqueeze(-1) - x2.unsqueeze(-2)
        x5 = x1.unsqueeze(-1) @ x2.unsqueeze(-2)
        # x4 = self.conv5(x4)
        # x5 = self.conv6(x5)
        x1 = self.tanh(torch.cat((x4, x5), dim=1))
        # x1 = self.tanh(x4)
        # x1, A = self.permutation(x1, A.unsqueeze(0).unsqueeze(0))
        x1 = self.conv4(x1) * alpha + A.unsqueeze(0).unsqueeze(0)
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
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
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)


        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model, self).__init__()
        source_path = os.path.abspath(__file__)
        shutil.copy2(source_path, "./logs/model.py")
        # if graph is None:
        #     raise ValueError()
        # else:
        #     Graph = import_class(graph)
        #     self.graph = Graph(**graph_args)
        #
        # A = self.graph.A # 3,25,25

        A = np.eye(25)
        adj_matrix = np.zeros((25, 25))
        adj_relation = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                        (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                        (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
        for i in adj_relation:
            adj_matrix[i[0] - 1, i[1] - 1] = 1
        A = np.stack((A, adj_matrix.T, adj_matrix), 0)  # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel * 2, A, stride=2, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel * 2, base_channel * 4, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive)

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