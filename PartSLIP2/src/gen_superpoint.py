"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
    Script for partioning into simples shapes
"""
import sys
import numpy as np
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir)
sys.path.append("%s/partition/cut-pursuit/build/src" % cur_dir.replace('src', ''))
#sys.path.append("%s/partition/ply_c" % cur_dir.replace('src', ''))
sys.path.append("%s/partition" % cur_dir.replace('src', ''))
import libcp
#import libply_c
from graphs import *
import open3d as o3d
from src.utils import save_colored_pc
from sklearn.neighbors import NearestNeighbors

import trimesh

def merge_small_components(components, xyz, min_component_size=10):
    n = len(components)
    large_component_xyz = []
    large_component_id = []
    for i in range(n):
        l = len(components[i])
        if l > min_component_size:
            large_component_xyz.append(xyz[components[i]])
            large_component_id.append([i] * l)
    large_component_xyz = np.concatenate(large_component_xyz)
    large_component_id = np.concatenate(large_component_id)
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(large_component_xyz)
    for i in range(n):
        if len(components[i]) <= min_component_size:
            _, neighbors = nn.kneighbors(xyz[components[i][0]].reshape(1, -1))
            best_comp_id = large_component_id[neighbors[0][0]]
            components[best_comp_id] += components[i]
            components[i] = []
    new_components = [com for com in components if len(com) > 0]
    return new_components

def visualize_superpoint(xyz, components, save_dir):
    rgb = np.zeros_like(xyz)
    for i in range(len(components)):
        rgb[components[i], :] = np.random.rand(3) * 255
    save_colored_pc(os.path.join(save_dir, "superpoint.ply"), xyz, rgb)

def gen_superpoint(xyz, rgb, normals, visualize=False, save_dir=None,
                   reg=0.05, k_nn_adj=10, k_nn_geof=45, lambda_edge_weight=1, mesh_path=None):
    #estimate point normals
    graph_nn, _ = compute_graph_nn_2(xyz, k_nn_adj, k_nn_geof)
    #geof = libply_c.compute_geof(xyz, target_fea, k_nn_geof).astype('float32')
    #del target_fea

    features = np.hstack((normals, rgb * 0.3)).astype('float32')
    feat_dis = np.linalg.norm(features[graph_nn["source"]] - features[graph_nn["target"]], axis = 1)
    graph_nn["edge_weight"] = np.array(1. / (lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"]) + feat_dis / feat_dis.mean() - 0.3), dtype = 'float32')
    graph_nn["edge_weight"] = graph_nn["edge_weight"] * graph_nn["edge_weight"]
    
    if mesh_path is not None:
        mesh = trimesh.load(mesh_path, force="mesh")
        adjacent_faces = trimesh.graph.face_adjacency(mesh=mesh)
        graph_nn["source"] = np.concatenate([graph_nn["source"], adjacent_faces[:,0]])
        graph_nn["target"] = np.concatenate([graph_nn["target"], adjacent_faces[:,1]])
        graph_nn["edge_weight"] = np.concatenate([graph_nn["target"], 0.4*np.ones_like(adjacent_faces[:,1])])

    components, _ = libcp.cutpursuit(features, graph_nn["source"], graph_nn["target"], graph_nn["edge_weight"], reg)

    components = merge_small_components(components, xyz)
    components = np.array(components, dtype = 'object')

    if visualize:
        visualize_superpoint(xyz, components, save_dir)
        np.save(f"{save_dir}/sp.npy", components)

    return components