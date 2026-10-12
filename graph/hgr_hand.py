import numpy as np
import sys

sys.path.extend(['../'])
from . import tools

num_node = 22
self_link = [(i, i) for i in range(num_node)]

inward_ori_index = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (21, 0),
    (21, 5),
    (21, 9),
    (21, 13),
    (21, 17),
]

inward = list(inward_ori_index)
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='uniform'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'uniform':
            A = tools.get_uniform_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'distance':
            A = tools.get_uniform_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os
    
    try:
        A = Graph('spatial').get_adjacency_matrix()
        print("Adjacency matrix shape:", A.shape)
        print("Number of partitions:", A.shape[0])
        
        graph_info = {
            'num_nodes': num_node,
            'num_edges': len(neighbor),
            'adjacency_shape': A.shape,
            'partitions': A.shape[0]
        }
        
        print("\nHand Graph Information:")
        for key, value in graph_info.items():
            print(f"{key}: {value}")
            
        print("\nHand keypoint connections:")
        print("Inward edges (parent -> child):", inward_ori_index[:10], "...")
        print("Total connections:", len(inward_ori_index))
        
    except Exception as e:
        print(f"Error creating graph: {e}")
        print("Please make sure graph/tools.py exists and is properly implemented")
