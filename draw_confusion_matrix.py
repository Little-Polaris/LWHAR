import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from Model import Model, Model5
from Model import ctrgcn
from plot_confusion_matrix import plot_confusion_matrix
from utils.MyDataLoader import MyDataLoader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, required=True, help='path of dataset preprocess configuration file ')
    args = vars(parser.parse_args())
    with open(args['config_path'], 'r') as file:
        config = dict(json.load(file))
    device = torch.device('cuda:0')

    weight = torch.load('logs/2025-04-10-19-44-22-perm-attn_cat/model_weight/model70.pth')
    # model = MyModel.Model().to(device).eval()
    model = Model5.Model(config['num_class'], config['num_point'],
                        config['num_person'], config['edges'],
                        config['dims']).to(device).eval()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_params}')
    model.load_state_dict(weight)
    train_data_loader, test_data_loader = MyDataLoader(config, device)

    del train_data_loader

    label_data = [i[1].numpy() for i in test_data_loader]
    label_data = np.concatenate(label_data, axis=0)

    y_pred = []
    test_data = torch.randn(1, 3, 64, 25, 2).to(device)
    for data in tqdm(test_data_loader):
        x, y, _ = data
        y_pred.append(model(x.to(device)).detach().cpu().numpy())


    # Convert the prediction result to class name
    y_pred = np.concatenate(y_pred, axis=0)
    y_pred = torch.tensor(np.array(y_pred)).squeeze().cuda()
    y_pred_label = list(torch.argmax(y_pred, dim=1).cpu())
    y_true_label = label_data  # [class_names[i] for i in np.argmax(y_test, axis=1)]

    # Plot the confusion matrix
    cm = confusion_matrix(label_data, torch.argmax(y_pred, dim=1).cpu())
    class_names = list(range(60))
    plot_confusion_matrix(cm,
                          class_names,
                          title='Confusion Matrix of Pose Classification Model')
    plt.show()

    # Print the classification report
    print('\nClassification Report:\n', classification_report(y_true_label,
                                                              y_pred_label, digits=16))
