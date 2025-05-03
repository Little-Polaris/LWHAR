import torch
from thop import profile

from Model import Model24

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_arg = {'Nh': 8, 'adjacency': False, 'agcn': False, 'all_layers': False, 'attention': True, 'attention_3': False, 'bn_flag': True, 'channel': 6, 'concat_original': True, 'data_normalization': True, 'dim_block1': 10, 'dim_block2': 30, 'dim_block3': 75, 'dk': 0.25, 'double_channel': True, 'drop_connect': True, 'dv': 0.25, 'graph': 'st_gcn.graph.NTU_RGB_D', 'graph_args': {'labeling_mode': 'spatial'}, 'kernel_temporal': 9, 'mask_learning': True, 'more_channels': False, 'n': 4, 'num_class': 60, 'num_person': 2, 'num_point': 25, 'only_attention': False, 'only_temporal_attention': True, 'relative': False, 'skip_conn': True, 'tcn_attention': True, 'use_data_bn': True, 'visualization': False, 'weight_matrix': 2, 'window_size': 300}
# model_arg = {'config': [[64, 64, 16], [64, 64, 16], [64, 128, 32], [128, 128, 32], [128, 256, 64], [256, 256, 64], [256, 256, 64], [256, 256, 64]], 'kernel_size': [3, 5], 'len_parts': 6, 'num_channels': 3, 'num_classes': 60, 'num_frames': 120, 'num_heads': 3, 'num_joints': 25, 'num_persons': 2, 'use_pes': True}
model = Model24.Model(60, 25, 2, 3).to(device)

flops, params = profile(model, inputs=(torch.randn(1, 3, 60, 25, 2).to(device)), verbose=False)
print(f'FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M')