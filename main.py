import argparse
import json
import os
import random
import warnings
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from torch import nn, autocast
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Model import ctrgcn, Model, MyModel, MyModel1, NewModel, temp
from utils.MyDataLoader import MyDataLoader
from utils.after_finish import after_finish
from utils.miniLogger import miniLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, required=True, help='path of dataset preprocess configuration file ')
    args = vars(parser.parse_args())
    with open(args['config_path'], 'r') as file:
        config = dict(json.load(file))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed = 1


    def init_seed(seed):
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    init_seed(seed)

    train_data_loader, test_data_loader = MyDataLoader(config, device)

    # 获取当前时间，用于创建日志文件夹
    start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    writer = SummaryWriter(f'logs/{start_time}')
    logger = miniLogger(start_time)
    logger.info(f'{config["dataset_name"]} {config["evaluation_mode"]}')

    # 创建模型
    model = Model.Model(config['num_class'], config['num_point'], config['num_person'], config['edges'], config['dims'])
    # model = MyModel.Model()
    # model = temp.Model()
    # model = MyModel1.Model()
    # model = NewModel.Model(config['num_class'], config['num_point'], config['dims'])
    model = model.to(device)
    logger.info(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total params: {total_params}')

    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    base_learning_rate = 0.01
    current_learning_rate = base_learning_rate
    lr_decay_rate = 0.1
    weight_decay = 0.0004

    optimizer = torch.optim.SGD(model.parameters(), lr=base_learning_rate, momentum=0.9, nesterov=True, weight_decay=weight_decay)

    last_lr_change_epoch = 0
    def adjust_learning_rate(epoch: int):
        learning_rate = 0.1
        if epoch < warm_up_epoch:
            lr = learning_rate * (epoch + 1) / warm_up_epoch
        else:
            lr = learning_rate * (
                    lr_decay_rate ** np.sum(epoch >= np.array([35, 55])))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

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

    scaler = GradScaler('cuda')
    # torch.autograd.set_detect_anomaly(True)

    for i in range(epoch + warm_up_epoch):
        if i > 4:
            current_learning_rate = adjust_learning_rate(i)
        # 训练模型
        model.train()
        temp = 0
        for data in tqdm(train_data_loader):
            temp+=1
            inputs, labels, _= data
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            with autocast('cuda'):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
            optimizer.zero_grad()

            scaler.scale(loss).backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            scaler.step(optimizer)
            scaler.update()

        # 测试训练效果
        cur_epoch_test_loss = 0
        model.eval()
        cur_acc = 0
        count = 0
        with torch.no_grad():
            for data in tqdm(test_data_loader):
                inputs, labels, _ = data
                inputs = inputs.float().to(device)
                labels = labels.long().to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                cur_epoch_test_loss += loss.item()
                count += 1
                cur_acc += (torch.argmax(outputs, 1) == labels).sum().item() / len(labels)
        # 记录训练日志并保存模型

        logger.info(f'Epoch {i + 1}, Loss: {cur_epoch_test_loss}, Acc: {cur_acc / count}, Learning Rate: {current_learning_rate}')
        acc.append(cur_acc / count)
        if (i + 1) % 5 == 0:
            state_dict = model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

            if not os.path.exists(f'./logs/{start_time}/model_weight'):
                os.makedirs(f'./logs/{start_time}/model_weight')
            torch.save(weights, f'./logs/{start_time}/model_weight/model{i + 1}.pth')
        writer.add_scalar('Loss/train', cur_epoch_test_loss, i + 1)
        writer.add_scalar('Acc/train', cur_acc / count, i + 1)
        if i == warm_up_epoch - 1:
            logger.info('Warm up finish, start training')

    after_finish(True)