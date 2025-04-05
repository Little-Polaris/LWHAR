import argparse
import json
import os
import random
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Model import ctrgcn
from utils.MyDataLoader import MyDataLoader
from utils.miniLogger import miniLogger

parser = argparse.ArgumentParser()
parser.add_argument('--config-path', type=str, required=True, help='path of dataset preprocess configuration file ')
args = vars(parser.parse_args())
with open(args['config_path'], 'r') as file:
    config = dict(json.load(file))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

train_data_loader, test_data_loader = MyDataLoader(config, device)

# 获取当前时间，用于创建日志文件夹
start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
writer = SummaryWriter(f'logs\\{start_time}')
logger = miniLogger(start_time)

# 创建模型
model = Model.Model()
# model = old_model.Trans(config['dims'], config['node_num'], config['edges'], config['action_class'])
model = model.to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f'Total params: {total_params}')
# print(f'Total params: {total_params}')

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

base_learning_rate = 0.1
current_learning_rate = base_learning_rate
lr_decay_rate = 0.1
weight_decay = 0.0004

# optimizer = torch.optim.SGD(model.parameters(), lr=base_learning_rate, momentum=0.9, nesterov=True, weight_decay=weight_decay)
optimizer = torch.optim.SGD(model.parameters(), lr=base_learning_rate)

last_lr_change_epoch = 0
def adjust_learning_rate(epoch: int, cur_lr: float):
    global last_lr_change_epoch
    if acc[epoch] - acc[epoch - 5] < 0.01 and epoch - last_lr_change_epoch > 5:
        cur_lr = cur_lr * lr_decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = cur_lr
        last_lr_change_epoch = epoch
    return cur_lr
    # if epoch < warm_up_epoch:
    #     lr = learning_rate * (epoch + 1) / warm_up_epoch
    # else:
    #     lr = learning_rate * (
    #             lr_decay_rate ** np.sum(epoch >= np.array([35, 55])))
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    # return lr

seed = 1
def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False          默认为PyTorch检测当前环境后自动处理
    # torch.backends.cudnn.deterministic = True     默认为False，可加快训练速度，当debug时可调整为True
    # torch.backends.cudnn.benchmark = False        默认为True，使得P有Torch可以自动寻找当前环境下的最优算法
init_seed(seed)

total_train_step = 0
total_test_step = 0
# 训练10000次
epoch = config['epoch']
warm_up_epoch = 0
if config.get('warm_up'):
    warm_up_epoch = config['warm_up']
    # print(f'Warm up for {warm_up_epoch} epoch')
    logger.info(f'Warm up for {warm_up_epoch} epoch')

acc = []

for i in range(epoch + warm_up_epoch):
    if i > 4:
        current_learning_rate = adjust_learning_rate(i, cur_lr=current_learning_rate)
    # 训练模型
    model.train()
    for data in tqdm(train_data_loader):
        inputs, labels = data
        # npinputs = inputs.cpu().numpy()
        # inputs = inputs.contiguous().reshape((24, 64, 2, 25, 3)).permute(0, 4, 1, 3, 2)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 测试训练效果
    cur_epoch_test_loss = 0
    model.eval()
    cur_acc = 0
    count = 0
    with torch.no_grad():
        for data in tqdm(test_data_loader):
            inputs, labels = data
            # inputs = inputs.contiguous().reshape((24, 64, 2, 25, 3)).permute(0, 4, 1, 3, 2)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            cur_epoch_test_loss += loss.item()
            count += 1
            cur_acc += (torch.argmax(outputs, 1) == labels).sum().item() / len(labels)
    # 记录训练日志并保存模型

    # logger.info(f'Epoch {i + 1}, Loss: {cur_epoch_test_loss}, Acc: {cur_acc / count}, Learning Rate: {current_learning_rate}')
    logger.info(f'Epoch {i + 1}, Loss: {cur_epoch_test_loss}, Acc: {cur_acc / count}, Learning Rate: {base_learning_rate}')
    acc.append(cur_acc / count)
    # torch.save(model, f'model\\model{i + 1}.pth')
    state_dict = model.state_dict()
    weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

    if not os.path.exists(f'model/model_weight'):
        os.makedirs(f'model/model_weight')
    torch.save(weights, f'model\\model_weight\\model{i + 1}.pth')
    writer.add_scalar('Loss/train', cur_epoch_test_loss, i + 1)
    writer.add_scalar('Acc/train', cur_acc / count, i + 1)
    if i == warm_up_epoch - 1:
        logger.info('Warm up finish, start training')
