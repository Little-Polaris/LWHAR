import argparse
import json
import os

import networkx as nx
import numpy as np
import plotly.graph_objs as go

parser = argparse.ArgumentParser()
parser.add_argument('--config-path', type=str, required=True, help='path of dataset preprocess configuration file ')
args = vars(parser.parse_args())
with open(args['config_path'], 'r') as file:
    config = dict(json.load(file))

file_name = list(os.listdir(config['raw_dataset_dir']))[0]
file = open(os.path.join(config['raw_dataset_dir'], file_name), 'r')
node_info = file.readlines()
node_info = node_info[4:4+int(config['node_num'])]
node_info = [i.strip().split() for i in node_info]
node_info = [i[:int(config['dims'])] for i in node_info]
node_info = np.array(node_info, np.float32)
node_info = [(i, {"pos":(node_info[i])}) for i in range(len(node_info))]

G = nx.Graph()
G.add_nodes_from(node_info)

if int(config['dims']) == 2:
    pos = nx.get_node_attributes(G, 'pos')
    x_nodes = [pos[k][0] for k in G.nodes()]
    y_nodes = [pos[k][1] for k in G.nodes()]

    x_edges = []
    y_edges = []

    for edge in G.edges():
        x_edges.extend([pos[edge[0]][0], pos[edge[1]][0], None])
        y_edges.extend([pos[edge[0]][1], pos[edge[1]][1], None])

    # 创建节点的散点图
    node_trace = go.Scatter(
        x=x_nodes, y=y_nodes,
        mode='markers+text',
        marker=dict(size=10, color='lightblue'),
        text=[str(node) for node in G.nodes()],
        textposition="top center"
    )

    # 创建边的线图
    # edge_trace = go.Scatter(
    #     x=x_edges, y=y_edges,
    #     mode='lines',
    #     line=dict(color='gray', width=2)
    # )

    # 创建图表布局
    layout = go.Layout(
        title='2D Interactive Graph',
        showlegend=False,
        xaxis=dict(title='X-axis'),
        yaxis=dict(title='Y-axis'),
        hovermode='closest'
    )

    # 创建图表
    fig = go.Figure(data=node_trace, layout=layout)
    fig.show()
else:
    pos = nx.get_node_attributes(G, 'pos')
    x_nodes = [pos[k][0] * 100 for k in G.nodes()]
    y_nodes = [pos[k][1] * 100 for k in G.nodes()]
    z_nodes = [pos[k][2] * 100 for k in G.nodes()]

    x_edges = []
    y_edges = []
    z_edges = []

    for edge in G.edges():
        x_edges.extend([pos[edge[0]][0], pos[edge[1]][0], None])
        y_edges.extend([pos[edge[0]][1], pos[edge[1]][1], None])
        z_edges.extend([pos[edge[0]][2], pos[edge[1]][2], None])

    # 创建节点的散点图
    node_trace = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers+text',
        marker=dict(size=10, color='lightblue'),
        text=[str(node) for node in G.nodes()],
        textposition="top center"
    )

    # 创建边的线图
    # edge_trace = go.Scatter3d(
    #     x=x_edges, y=y_edges, z=z_edges,
    #     mode='lines',
    #     line=dict(color='gray', width=2)
    # )

    # 创建图表布局
    layout = go.Layout(
        title='3D Interactive Graph',
        showlegend=False,
        scene=dict(
            xaxis=dict(title='X-axis'),
            yaxis=dict(title='Y-axis'),
            zaxis=dict(title='Z-axis')
        )
    )

    # 创建图表
    fig = go.Figure(data=node_trace, layout=layout)
    fig.show()



