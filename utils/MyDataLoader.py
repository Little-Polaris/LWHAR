import os
import pickle
import random

import numpy as np
import torch
from scipy.interpolate import interp1d

from torch.utils.data import Dataset


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class Feeder(Dataset):
    def  __init__(self, data_path, p_interval, split, evaluation_mode, window_size, edges,
                  bone=0, vel=0):
        self.data_path = data_path
        self.p_interval = p_interval
        self.split = split
        self.evaluation_mode = evaluation_mode
        self.window_size = window_size
        self.edges = edges
        self.bone = bone
        self.vel = vel
        self.load_data()

    def load_data(self):
        npz_data = pickle.load(open(self.data_path, 'rb'))
        if self.split == 'train':
            self.data = npz_data[f'{self.evaluation_mode}_train_data']['data']
            self.label = npz_data[f'{self.evaluation_mode}_train_data']['label']
            # self.sample_name = npz_data[f'{self.evaluation_mode}_train_data']['filename']
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data[f'{self.evaluation_mode}_test_data']['data']
            self.label = npz_data[f'{self.evaluation_mode}_test_data']['label']
            # self.sample_name = npz_data[f'{self.evaluation_mode}_test_data']['filename']
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        data = self.valid_crop_resize(data, data.shape[1], self.p_interval, self.window_size)
        data = self.rot(data)
        if self.bone:
            bone_data = torch.zeros_like(data)
            for v1, v2 in self.edges:
                bone_data[:, :, v1 - 1] = data[:, :, v1 - 1] - data[:, :, v2 - 1]

            data = bone_data
        if self.vel:
            data = data[:, 1:] - data[:, :-1]
            data[:, -1] = 0
        return data, label, index

    def interpolate_isolated_zeros_in_columns(self, data: np.ndarray[np.float32], window_size: int) -> np.ndarray[np.float32]:
        C, T, V, M = data.shape
        data = data.transpose(1, 2, 3, 0).reshape(T, V * M * C)
        magnification = window_size // T
        remainder = window_size % T
        interval = np.ones(T, dtype=np.int32) * magnification

        interval[np.random.choice(T, remainder, replace=False)] += 1
        j = 0
        interval_index = 0
        interpolated_data = np.zeros((window_size, V * C * M), dtype=np.float32)
        while j < window_size:
            interpolated_data[j, :V * C * M] = data[interval_index]
            j += interval[interval_index]
            interval_index += 1
        non_zero_index = np.where(~np.all(interpolated_data == 0, axis=1))[0]
        interp_func = interp1d(non_zero_index, interpolated_data[non_zero_index, :], kind='linear', axis=0,
                               fill_value="extrapolate")
        interpolated_data[:window_size] = interp_func(np.arange(window_size))
        interpolated_data = interpolated_data.reshape(window_size, V, M, C).transpose(3, 0, 1, 2)
        return interpolated_data

    def valid_crop_resize(self, data_numpy, valid_frame_num, p_interval, window_size):
        begin = 0
        end = valid_frame_num
        valid_size = end - begin

        if len(p_interval) == 1:
            p = p_interval[0]
            bias = int((1 - p) * valid_size / 2)
            data = data_numpy[:, begin + bias:end - bias, :, :]  # center_crop
        else:
            p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
            cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p.item())), window_size),
                                        valid_size)
            bias = np.random.randint(0, valid_size - cropped_length + 1)
            data = data_numpy[:, begin + bias:begin + bias + cropped_length, :, :]
            if data.shape[1] == 0:
                print(cropped_length, bias, valid_size)

        data = self.interpolate_isolated_zeros_in_columns(data, window_size)

        return torch.tensor(data)

    def rot_aux(self, rot):
        if len(rot.shape) == 2 and rot.shape[1] == 2:
            cos_r, sin_r = rot.cos(), rot.sin()
            r1 = torch.stack((cos_r[:, 0], -sin_r[:, 0]), dim=-1)
            r2 = torch.stack((sin_r[:, 0], cos_r[:, 0]), dim=-1)
            rot = torch.stack((r1, r2), dim=1)  # T,2,2
        elif len(rot.shape) == 2 and rot.shape[1] == 3:
            cos_r, sin_r = rot.cos(), rot.sin()  # T,3
            zeros = torch.zeros(rot.shape[0], 1)  # T,1
            ones = torch.ones(rot.shape[0], 1)  # T,1

            r1 = torch.stack((ones, zeros, zeros), dim=-1)  # T,1,3
            rx2 = torch.stack((zeros, cos_r[:, 0:1], sin_r[:, 0:1]), dim=-1)  # T,1,3
            rx3 = torch.stack((zeros, -sin_r[:, 0:1], cos_r[:, 0:1]), dim=-1)  # T,1,3
            rx = torch.cat((r1, rx2, rx3), dim=1)  # T,3,3

            ry1 = torch.stack((cos_r[:, 1:2], zeros, -sin_r[:, 1:2]), dim=-1)
            r2 = torch.stack((zeros, ones, zeros), dim=-1)
            ry3 = torch.stack((sin_r[:, 1:2], zeros, cos_r[:, 1:2]), dim=-1)
            ry = torch.cat((ry1, r2, ry3), dim=1)

            rz1 = torch.stack((cos_r[:, 2:3], sin_r[:, 2:3], zeros), dim=-1)
            r3 = torch.stack((zeros, zeros, ones), dim=-1)
            rz2 = torch.stack((-sin_r[:, 2:3], cos_r[:, 2:3], zeros), dim=-1)
            rz = torch.cat((rz1, rz2, r3), dim=1)

            rot = rz.matmul(ry).matmul(rx)
        return rot

    def rot(self, data, theta=0.3):
        """
        data_numpy: C,T,V,M
        """
        C, T, V, M = data.shape
        data = data.permute(1, 0, 2, 3).contiguous().view(T, C, V * M)  # T,3,V*M
        rot = torch.zeros(C).uniform_(-theta, theta)
        rot = torch.stack([rot, ] * T, dim=0)
        rot = self.rot_aux(rot)  # T,3,3
        data = torch.matmul(rot, data)
        data = data.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()

        return data

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def MyDataLoader(config, device):
    train_data_loader = torch.utils.data.DataLoader(
                dataset=Feeder(data_path=os.path.join(config['binary_data_dir'], f'{config["dataset_name"]}_{config["evaluation_mode"]}_processed_data.pkl'),
                               p_interval=config['train_p_interval'],
                               split='train',
                               evaluation_mode=config['evaluation_mode'],
                               window_size=64,
                               edges=config['edges'],
                               bone=config['bone'],
                               vel=config['vel']),
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=config['dataloader_num_workers'],
                prefetch_factor=config['dataloader_num_workers'] if config['dataloader_num_workers'] > 0 else None,
                drop_last=True,
                worker_init_fn=init_seed)

    test_data_loader = torch.utils.data.DataLoader(
            dataset=Feeder(data_path=os.path.join(config['binary_data_dir'], f'{config["dataset_name"]}_{config["evaluation_mode"]}_processed_data.pkl'),
                           p_interval=config['test_p_interval'],
                           split='test',
                           evaluation_mode=config['evaluation_mode'],
                           window_size=64,
                           edges=config['edges'],
                           bone=config['bone'],
                           vel=config['vel']),
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['dataloader_num_workers'],
            prefetch_factor=config['dataloader_num_workers'] if config['dataloader_num_workers'] > 0 else None,
            drop_last=False,
            worker_init_fn=init_seed)
    return train_data_loader, test_data_loader