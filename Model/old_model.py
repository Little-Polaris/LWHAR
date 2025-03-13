import torch
from torch import nn


class DotProductAttention(nn.Module):
    '''
    点积注意力机制
        通过计算查询向量和键向量的点积，得到不同关节之间的注意力权重，从而实现对不同关节的关注。
        同时，通过添加邻接矩阵，引入了图网络结构，使得模型能够学习到关节之间的关系。
    '''
    def __init__(self, node_num: int, edges: list[list[int]]):
        super(DotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.edges = edges
        # self.edges = torch.tensor(
        #     [[0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 6], [5, 11], [6, 12],
        #      [11, 12], [11, 13], [13, 15], [12, 14]])
        self.adjacent_matrix = torch.zeros((64, node_num * 2, 64, node_num * 2)).to('cuda' if torch.cuda.is_available() else 'cpu')
        # self.adjacent_matrix = torch.eye(node_num).to('cuda' if torch.cuda.is_available() else 'cpu')
        for i in self.edges:
            self.adjacent_matrix[:, i[0] - 1, :, i[1] - 1] = 1
            self.adjacent_matrix[:, i[0] - 1 + node_num, :, i[1] - 1 + node_num] = 1
            self.adjacent_matrix[:, i[1] - 1, :, i[0] - 1] = 1
            self.adjacent_matrix[:, i[1] - 1 + node_num, :, i[0] - 1 + node_num] = 1

        self.adjacent_matrix = nn.Parameter(self.adjacent_matrix, requires_grad=True)

    def forward(self, q, k):
        N, T, V, C = q.shape
        q = q.reshape(N, T * V, C)
        k = k.reshape(N, T * V, C)
        x = torch.matmul(q, k.transpose(1, 2))
        # x = q @ k.transpose(1, 2)
        x = x.reshape(N, T, V, T, V)
        x += self.adjacent_matrix
        x = x.reshape(N, T * V, T * V)
        x = self.softmax(x)
        return x


class Attention(nn.Module):
    '''
    注意力模块
        通过注意力机制，实现对不同关节的关注，再与原始输入相乘，得到各个关节的语义信息
    '''
    def __init__(self, dims: int, node_num: int, edges: list[list[int]]):
        super(Attention, self).__init__()
        self.W_K = nn.Conv2d(dims, dims * 4, 1)
        self.W_Q = nn.Conv2d(dims, dims * 4, 1)

        self.attention = DotProductAttention(node_num, edges)

    def forward(self, x: torch.Tensor):
        q = self.W_Q(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        k = self.W_K(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # q = self.W_Q(x.transpose(1, 2)).transpose(1, 2)
        # k = self.W_K(x.transpose(1, 2)).transpose(1, 2)
        v = x

        x = self.attention(q, k)
        x = torch.matmul(x, v)
        return x


class FeedForwardNet(nn.Module):
    '''
    前馈神经网络
        通过卷积层和Dropout层，实现对语义信息的提取和降维
    '''
    def __init__(self, dims: int):
        super(FeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(dims, dims * 4, 1)
        self.drop1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(dims * 4, dims, 1)
        self.drop2 = nn.Dropout(0.2)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.drop2(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    '''
    编码器层
        通过注意力模块和前馈神经网络，实现对输入数据的特征提取和降维
    '''
    def __init__(self, dims: int, node_num: int, edges: list[list[int]]):
        super(EncoderLayer, self).__init__()
        self.attention = Attention(dims, node_num, edges)
        self.feed_forward = FeedForwardNet(dims)

    def forward(self, x):
        x = self.attention(x)
        x = self.feed_forward(x)
        return x


class Encoder(nn.Module):
    '''
    编码器
        通过编码器层，实现对输入数据的特征提取和降维。
        由于模型最终会被部署到移动设备，因此编码器层的数量较少，以减少模型的大小和计算量。
    '''
    def __init__(self, dims: int, node_num: int, edges: list[list[int]]):
        super(Encoder, self).__init__()
        self.layer = EncoderLayer(dims, node_num, edges)

    def forward(self, x):
        x = self.layer(x)
        return x


class Trans(nn.Module):
    '''
    模型
        通过编码器，实现对输入数据的特征提取和降维，再通过全连接层，实现对特征的分类。
    '''
    def __init__(self, dims: int, node_num: int, edges: list[list[int]], action_class: int):
        super(Trans, self).__init__()
        self.encoder = Encoder(dims, node_num, edges)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(dims * node_num, 64)
        self.act1 = nn.ReLU6()
        self.drop1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(64, action_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
