import math

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

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=pad,
            stride=(stride, 1),
            dilation=dilation)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Permutation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.permutation_matrix = nn.Parameter(torch.arange(dim, dtype=torch.float32))

    def forward(self, x):
        permutation_matrix = self.permutation_matrix.to(torch.long)
        permutation_matrix = permutation_matrix.reshape(1, 1, 1, -1)
        permutation_matrix = permutation_matrix.repeat(x.shape[0], x.shape[1], x.shape[2], 1)
        x_permuted = torch.gather(x, 3, permutation_matrix)
        return x_permuted

class st_feature_extraction_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, pooling=None):
        super(st_feature_extraction_block, self).__init__()
        self.stride = stride if isinstance(stride, tuple) else (stride, 1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        if pooling == 'max':
            self.pooling = nn.MaxPool2d(kernel_size=kernel_size, stride=self.stride, padding=padding)
            self.stride = 1
        elif pooling == 'avg':
            self.pooling = nn.AvgPool2d(kernel_size=kernel_size, stride=self.stride, padding=padding)
            self.stride = 1
        else:
            self.pooling = return_x()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=self.stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pooling(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, channels, time_length, num_point, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(time_length, channels)
        position = torch.arange(0, time_length, dtype=torch.float).unsqueeze(1)  # 生成位置索引
        div_term = torch.exp(torch.arange(0, channels, 2).float() * (-math.log(10000.0) / channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2).unsqueeze(-1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)


class st_feature_extraction(nn.Module):
    def __init__(self , in_channels, out_channels, num_point, time_length, stride=1):
        super(st_feature_extraction, self).__init__()
        self.num_point = num_point
        self.pe = PositionalEncoding(in_channels, time_length, num_point)
        self.permutations = nn.ModuleList([Permutation(self.num_point) for i in range(4)])
        self.block1 = st_feature_extraction_block(in_channels, out_channels // 4, 3, stride, 1, 1)
        self.block2 = st_feature_extraction_block(in_channels, out_channels // 4, 3, stride, 2, 2)
        self.block3 = st_feature_extraction_block(in_channels, out_channels // 4, 1, stride, 0, 1, 'max')
        self.block4 = st_feature_extraction_block(in_channels, out_channels // 4, 1, stride, 0, 1, 'avg')
        self.apply(weights_init)

    def forward(self, x):
        x = self.pe(x)
        x_permuted = self.permutations[0](x)
        out1 = self.block1(x_permuted)
        x_permuted = self.permutations[1](x)
        out2 = self.block2(x_permuted)
        x_permuted = self.permutations[2](x)
        out3 = self.block3(x_permuted)
        x_permuted = self.permutations[3](x)
        out4 = self.block4(x_permuted)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        return out

class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 time_length,
                 num_point,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        # self.num_branches = len(dilations)
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        self.time_length = time_length
        self.num_point = num_point
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
            nn.MaxPool2d(kernel_size=3, stride=(stride, 1), padding=1),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.permutations = nn.ModuleList([Permutation(self.num_point) for i in self.branches])
        # Residual connection
        if not residual:
            # self.residual = lambda x: 0
            self.residual = return_0()
        elif (in_channels == out_channels) and (stride == 1):
            # self.residual = lambda x: x
            self.residual = return_x()
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for i in range(len(self.branches)):
            x_permuted = self.permutations[i](x)
            out = self.branches[i](x_permuted)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out

class dot_product_attention(nn.Module):
    def __init__(self):
        super(dot_product_attention, self).__init__()
        self.tanh = nn.Tanh()
        self.alpha = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))

    def forward(self, k, q):
        x1 = k.unsqueeze(-1) @ q.unsqueeze(-2)
        x2 = k.unsqueeze(-1) - q.unsqueeze(-2)
        x = x1 * self.alpha + x2 * self.beta
        x = self.tanh(x)
        return x

class st_attention(nn.Module):
    def __init__(self, in_channels, out_channels, num_point, rel_reduction=8, mid_reduction=1):
        super(st_attention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.W_K = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.W_Q = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.W_V = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.dot_production_attention = dot_product_attention()
        self.ffn = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor([0], dtype=torch.float32))
        if num_point == 25:
            self.edges = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                          (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                          (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                          (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
        elif num_point == 17:
            self.edges = torch.tensor(
                [[1, 2], [1, 3], [2, 4], [3, 5], [1, 6], [1, 7], [6, 8], [8, 10], [7, 9], [9, 11], [6, 7], [6, 12],
                 [7, 13],
                 [12, 13], [12, 14], [14, 16], [13, 15]])
        self.adjc_mat = torch.eye(num_point).to('cuda' if torch.cuda.is_available() else 'cpu')
        for i in self.edges:
            self.adjc_mat[i[0] - 1][i[1] - 1] = 1
            self.adjc_mat[i[1] - 1][i[0] - 1] = 1
        self.adjc_mat = nn.Parameter(self.adjc_mat, requires_grad=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x):
        k = self.W_K(x).mean(-2)
        q = self.W_Q(x).mean(-2)
        v = self.W_V(x)

        x = self.dot_production_attention(k, q)

        x = self.ffn(x)

        x = self.alpha * x + self.adjc_mat.unsqueeze(0).unsqueeze(0)

        x = x @ v.transpose(-1, -2)
        x = x.transpose(-1, -2).contiguous()
        return x

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
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


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, num_point, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = 3
        # self.CTRGC = CTRGC(in_channels, out_channels)
        self.attention = nn.ModuleList()
        for i in range(self.num_subset):
            self.attention.append(st_attention(in_channels, out_channels, num_point))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                # self.down = lambda x: x
                self.down = return_x()
        else:
            # self.down = lambda x: 0
            self.down = return_0()
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
        for i in range(self.num_subset):
            z = self.attention[i](x)
            y = z + y if y is not None else z
        # y = self.CTRGC(x, A)
        y = self.bn(y)
        z = self.down(x)
        y = torch.add(y, z)
        y = self.relu(y)


        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, num_point, time_length, stride=1, residual=True, adaptive=True, kernel_size=3, dilations=[1, 2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, num_point, adaptive=adaptive)
        # self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, time_length, num_point, kernel_size=kernel_size, stride=stride, dilations=dilations,
        #                                     residual=False)
        self.tcn1 = st_feature_extraction(out_channels, out_channels, num_point, time_length, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            # self.residual = lambda x: 0
            self.residual = return_0()

        elif (in_channels == out_channels) and (stride == 1):
            # self.residual = lambda x: x
            self.residual = return_x()

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=(stride, 1))

    def forward(self, x):
        x1 = self.gcn1(x)
        x2 = self.tcn1(x1)
        res = self.residual(x)
        y = self.relu(x2 + res)
        # y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model, self).__init__()


        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        # self.data_bn2 = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        time_length = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, num_point, time_length, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, num_point, time_length, adaptive=adaptive)
        # self.l3 = TCN_GCN_unit(base_channel, base_channel, A, time_length, num_point, adaptive=adaptive)
        # self.l4 = TCN_GCN_unit(base_channel, base_channel, A, time_length, num_point, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, num_point, time_length, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, num_point, time_length//2,  adaptive=adaptive)
        # self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, time_length//2, num_point, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, num_point, time_length//2, stride=2, adaptive=adaptive)
        # self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, time_length//4, num_point, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, num_point, time_length//4, adaptive=adaptive)

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        # bn_init(self.data_bn2, 1)
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            # self.drop_out = lambda x: x
            self.drop_out = return_x()

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        # is_one_person = torch.all(x == 0, dim=2)[:, 75]
        # one_person_index = torch.nonzero(is_one_person).squeeze()
        # two_person_index = torch.nonzero(~is_one_person).squeeze()
        # x1 = x[one_person_index].reshape(-1, M*V*C, T)
        # x2 = x[two_person_index].reshape(-1, M*V*C, T)
        # if one_person_index.numel() > 1:
        #     x1 = self.data_bn1(x1)
        # if two_person_index.numel() > 1:
        #     x2 = self.data_bn2(x2)
        # normalized_x = torch.zeros_like(x)
        # normalized_x[one_person_index] = x1
        # normalized_x[two_person_index] = x2
        normalized_x = self.data_bn(x)
        x = normalized_x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        # x = self.l3(x)
        # x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        # x = self.l7(x)
        x = self.l8(x)
        # x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
        x = self.fc(x)

        return x