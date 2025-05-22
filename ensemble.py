import argparse
import json
import os
import pickle
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count
from sklearn.metrics import confusion_matrix, classification_report
from thop import profile
from tqdm import tqdm

from Model import Model, Model12, Model13, Model5
from Model import ctrgcn
from Model import Model24
from plot_confusion_matrix import plot_confusion_matrix
from utils.MyDataLoader import MyDataLoader
from utils.miniLogger import miniLogger

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, required=True, help='path of evaluate configuration file ')
    args = vars(parser.parse_args())
    with open(args['config_path'], 'r') as file:
        config = dict(json.load(file))
    device = torch.device(config['device'])

    joint_weight = torch.load(f'{config["joint_weight_path"]}')
    joint_motion_weight = torch.load(f'{config["joint_motion_weight_path"]}')
    bone_weight = torch.load(f'{config["bone_weight_path"]}')
    bone_motion_weight = torch.load(f'{config["bone_motion_weight_path"]}')
    weights = [joint_weight, joint_motion_weight, bone_weight, bone_motion_weight]
    weights_config = [{'bone':0, 'vel':0},
                      {'bone':0, 'vel':1},
                      {'bone':1, 'vel':0},
                      {'bone':1, 'vel':1}]
    
    model = Model24.Model(config['num_class'], config['num_point'],
                        config['num_person'], config['dims']).to(device).eval()

    start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.mkdir(f'./logs/{start_time}')
    logger = miniLogger(start_time)
    shutil.copy2(args['config_path'], f'logs/{start_time}/config.json')
    shutil.move('./logs/model.py', f"./logs/{start_time}/model.py")
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # logger.info(f'Total params: {total_params}')


    label_map_file = open('./lables.txt')
    label_map = label_map_file.readlines()
    label_map = {i.split()[0][1:]: i.split()[1] for i in label_map}
    label_map_file.close()
    label_names = list(label_map.values())
    if config['dataset_name'] == 'ntu60':
        label_names = label_names[:60]

    outputs = [[], [], [], []]
    config['bone'] = 0
    config['vel'] = 0
    _, test_data_loader = MyDataLoader(config, device)
    true_lables = numpy.array(test_data_loader.dataset.label, dtype=int)

    for i in range(4):
        model.load_state_dict(weights[i], strict=False)
        model.eval()
        config['bone'] = weights_config[i]['bone']
        config['vel'] = weights_config[i]['vel']
        _, test_data_loader = MyDataLoader(config, device)
        true_lables = numpy.array(test_data_loader.dataset.label, dtype=int)
        flops, params = profile(model, inputs=(torch.randn(1, 3, 64, 25, 2).to(device), ), verbose=False)
        logger.info(f'FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M')
        for data, _, _ in tqdm(test_data_loader):
            data = data.float().to(device)
            outputs[i].extend(model(data).detach().cpu().numpy().tolist())
    ratio = [0.6, 0.4, 0.6, 0.4]
    outputs = [torch.tensor(outputs[i]) for i in range(4)]
    result = outputs[0] * ratio[0] + outputs[1] * ratio[1] + outputs[2] * ratio[2] + outputs[3] * ratio[3]
    result = torch.argmax(result, dim=1).numpy()
    # with open('./result.pkl', 'rb') as f:
    #     result = pickle.load(f)
    cm = confusion_matrix(true_lables, result)
    plot_confusion_matrix(cm,
                          [i for i in range(1, len(label_names) + 1)],
                          title='Confusion Matrix of Pose Classification Model')
    # plot_confusion_matrix(cm,
    #                       label_names,
    #                       title='Confusion Matrix of Pose Classification Model')

    plt.savefig(f'./logs/{start_time}/confusion_matrix.pdf', format='pdf', bbox_inches="tight")
    # plt.show()

    # Print the classification report
    logger.info(classification_report(true_lables, result, digits=4))

    pass