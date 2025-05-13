import os
import re
import json
import yaml
import scipy
import trimesh
import numpy as np
import open3d as o3d

from parse_utils import parse_articulation_tree


def pointcloud_distance(pc_a, pc_b):
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc_a)
    pcd2.points = o3d.utility.Vector3dVector(pc_b)

    dist = np.asarray(pcd2.compute_point_cloud_distance(pcd1)) 

    return dist


def postprocess_segmentation(config):

    data_dir = os.path.join(config['data_dir'], config['object_name'])
    use_cc = config["segmentation"]["use_cc"]
    cc_threshold = config["segmentation"]["cc_threshold"]

    mesh = trimesh.load(os.path.join(data_dir, "mesh_normalized.glb"), force="mesh")
    num_faces = mesh.faces.shape[0]
    num_points_per_face = np.zeros(num_faces)
    points_face_idx = np.load(os.path.join(data_dir, "points_face_idx.npy"))
    used_pts = np.zeros(points_face_idx.shape[0])
    face_indices, count = np.unique(points_face_idx, return_counts=True)
    for idx, face_idx in enumerate(face_indices):
        num_points_per_face[face_idx] += count[idx]

    ### connected components for structured mesh
    if use_cc:
        trimesh.grouping.merge_vertices(mesh, merge_tex=True, merge_norm=True)
        face_adjacency = mesh.face_adjacency
        face_adjacency = np.concatenate([face_adjacency, np.array([[mesh.faces.shape[0]-1, mesh.faces.shape[0]-1]])])
        cc_labels = trimesh.graph.connected_component_labels(face_adjacency)

        num_cc_labels = len(np.unique(cc_labels))
        if num_cc_labels == 1 or num_cc_labels > 1000:
            use_cc = False

        point_cc_label = cc_labels[points_face_idx]

    ### Extract each part from the mesh
    os.makedirs(os.path.join(data_dir, "parts"), exist_ok=True)
    instance_seg_path = os.path.join(data_dir, "instance_seg")
    link_name_txt = ""
    link_parent_txt = ""
    num_links = -1
    part_idx_dict = {}
    articulation_tree = parse_articulation_tree(os.path.join(data_dir, "gpt4o_out", "articulation_tree.txt"))

    for part_idx, sem_part in enumerate(os.listdir(instance_seg_path)):
        instance_seg_pc = o3d.io.read_point_cloud(os.path.join(instance_seg_path, sem_part))
        for ins_idx, ins_rgb in enumerate(np.unique(np.asarray(instance_seg_pc.colors), axis=0)):
            if (ins_rgb == 0.).all():
                continue
            part_link_name = sem_part.replace('.ply', '')

            num_selected_points_per_face = np.zeros(num_faces)
            instance_id = np.asarray(instance_seg_pc.colors)
            selected_pts_indices = np.abs(np.sum(instance_id - ins_rgb.reshape(1,3), axis=-1)) < 1e-5
            if use_cc:
                selected_pts_cc_labels = point_cc_label[selected_pts_indices]
                cc_indices = []
                for cc_idx in range(len(np.unique(cc_labels))):
                    pcd1 = np.asarray(instance_seg_pc.points)[selected_pts_indices]
                    pcd2 = np.asarray(instance_seg_pc.points)[point_cc_label==cc_idx]
                    dist = pointcloud_distance(pcd1, pcd2) # only distance from cc to selected pts

                    if np.mean(dist) < cc_threshold:
                        cc_indices.append(cc_idx)
                
                if len(cc_indices) == 0:
                    continue
                else:
                    selected_pts_indices = np.sum(np.stack([point_cc_label == i for i in cc_indices], axis=0), axis=0) > 0

            used_pts[selected_pts_indices] = 1
            selected_pts = np.asarray(instance_seg_pc.points)[selected_pts_indices]
            np.save(os.path.join(data_dir, "parts", f"{part_link_name}@{ins_idx}.npy"), selected_pts)
            num_links += 1
            part_idx_dict[f"{part_link_name}@{ins_idx}"] = num_links

            selected_faces = points_face_idx[selected_pts_indices]
            selected_face_indices, count = np.unique(selected_faces, return_counts=True)
            for idx, face_idx in enumerate(selected_face_indices):
                num_selected_points_per_face[face_idx] += count[idx]
            selected_ratio = num_selected_points_per_face / num_points_per_face
            selected_ratio = np.nan_to_num(selected_ratio)
            selected_faces = selected_ratio > 0.5
            mesh_tmp = trimesh.load(os.path.join(data_dir, "mesh_normalized.glb"), force="mesh")
            mesh_tmp.update_faces(selected_faces)
            mesh_tmp.remove_unreferenced_vertices()
            os.makedirs(os.path.join(data_dir, "mesh_parts"), exist_ok=True)
            mesh_tmp.export(os.path.join(data_dir, "mesh_parts", sem_part.replace('.ply', f'@{ins_idx}.glb')))

            link_name_txt += f"{num_links}. {part_link_name}@{ins_idx}\n"

    ### faces that does not belong to any movable part are the base
    all_pc = o3d.io.read_point_cloud(os.path.join(data_dir, "sampled_points.ply"))
    base_pts = np.asarray(instance_seg_pc.points)[used_pts == 0]
    np.save(os.path.join(data_dir, "parts", f"base.npy"), base_pts)
    link_name_txt += f"{num_links+1}. base\n"
    selected_faces = points_face_idx[used_pts == 0]
    selected_face_indices, count = np.unique(selected_faces, return_counts=True)
    num_selected_points_per_face = np.zeros(num_faces)
    for idx, face_idx in enumerate(selected_face_indices):
        num_selected_points_per_face[face_idx] += count[idx]
    selected_ratio = num_selected_points_per_face / num_points_per_face
    selected_ratio = np.nan_to_num(selected_ratio)
    selected_faces = selected_ratio > 0.5
    mesh_tmp = trimesh.load(os.path.join(data_dir, "mesh_normalized.glb"), force="mesh")
    mesh_tmp.update_faces(selected_faces)
    mesh_tmp.remove_unreferenced_vertices()

    os.makedirs(os.path.join(data_dir, "mesh_parts"), exist_ok=True)
    mesh_tmp.export(os.path.join(data_dir, "mesh_parts", "base.glb"))

    ### find parent link
    for part_i in part_idx_dict.keys():
        part_link_name = part_i.split("@")[0]
        parent_link = articulation_tree[part_link_name]['parent']
        part_i_pts = np.load(os.path.join(data_dir, "parts", f"{part_i}.npy"))
        min_dist = 10000
        parent_part = None

        if parent_link == 'base':
            link_parent_txt += f"{part_idx_dict[part_i]}. base\n"
            continue

        for part_j in part_idx_dict.keys():
            if parent_link in part_j:
                part_j_pts = np.load(os.path.join(data_dir, "parts", f"{part_j}.npy"))
                dist = pointcloud_distance(part_j_pts, part_i_pts)
                
                if dist < min_dist:
                    min_dist = dist
                    parent_part = part_j
        
        link_parent_txt += f"{part_idx_dict[part_i]}. {parent_part}\n"

    link_name_txt = "```link_names\n" + link_name_txt + "```"
    link_name_txt += "\n\n```parent_links\n" + link_parent_txt + "```"
    with open(os.path.join(data_dir, "gpt4o_out", "link_hierarchy.txt"), 'w') as f:
        f.write(link_name_txt)