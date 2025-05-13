import argparse
import os
import re
import cv2
import copy
import json
import math
import trimesh
import numpy as np
import networkx as nx
import open3d as o3d

from sklearn.cluster import DBSCAN, KMeans
from collections import Counter

from render_object import get_camera_poses
from parse_utils import parse_articulation_tree, parse_link_hierarchy

def calculate_camera_intrinsics(image_width, image_height, xfov_degrees, yfov_degrees):
    xfov_radians = np.radians(xfov_degrees)
    yfov_radians = np.radians(yfov_degrees)
    fx = image_width / (2 * np.tan(xfov_radians / 2))
    fy = image_height / (2 * np.tan(yfov_radians / 2))
    cx = image_width / 2
    cy = image_height / 2
    
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ])
    
    return K

def project_pcd(w2c, pcd, K):
    pts_homogenous = np.concatenate([pcd, np.ones((pcd.shape[0],1))], axis=-1)
    K_34 = np.zeros((3,4))
    K_34[:3,:3] = K
    projection_matrix = K_34 @ w2c
    pts_cam = pts_homogenous @ projection_matrix.T
    pts_cam /= pts_cam[:, 2:]
    pts_cam = pts_cam.astype(int)

    return pts_cam

def filter_outliers_DBSCAN(xyz, eps=0.02):
    db = DBSCAN(eps=eps, min_samples=5)
    db.fit(xyz)
    labels = db.labels_
    label_counts = Counter(labels)
    most_common_label, most_common_count = label_counts.most_common(1)[0]
    mask = labels == most_common_label
    xyz = xyz[mask]

    return xyz

def compute_contact_points(pcd_part, pcd_rest):
    threshold = 0.01
    while True:
        comp0_tree = o3d.geometry.KDTreeFlann(pcd_rest)
        contact_points_part = []
        bitmap = np.zeros(np.asarray(pcd_rest.points).shape[0])
        for point in pcd_part.points:
            [_, idx, distances] = comp0_tree.search_radius_vector_3d(point, threshold)
            if len(idx) != 0:
                contact_points_part.append(np.asarray(point))
            for pt_idx in idx:
                bitmap[pt_idx] = 1

        contact_points_rest = np.asarray(pcd_rest.points)[bitmap == 1]

        if len(contact_points_part) != 0:
            break
        else:
            threshold *= 2

    contact_points_part = np.stack(contact_points_part, axis=0)
    contact_points = np.concatenate([contact_points_part, contact_points_rest], axis=0)

    return contact_points, contact_points_rest

def cluster_contact_points(contacts, w2c, K, config, zoom_ratio):
    final_middle_points = None
    final_contact_points_color = None
    for num_clusters in range(1,21):
        km = KMeans(n_clusters=num_clusters, random_state=0)
        km.fit(contacts)
        labels = km.labels_

        middle_points = []
        contact_points_color = np.zeros_like(contacts)
        for lb in np.unique(labels):
            if lb == -1:
                continue
            xyz = contacts[labels==lb]
            middle_point = np.mean(xyz, axis=0)
            middle_points.append(middle_point)
            contact_points_color[labels==lb] = np.random.rand(3)*np.ones((xyz.shape[0], 3)) * 255

        middle_points = np.stack(middle_points, axis=0)

        middle_pts_cam = project_pcd(w2c, middle_points, K)

        pts_dist = []
        for idx_i in range(middle_pts_cam.shape[0]):
            for idx_j in range(idx_i+1, middle_pts_cam.shape[0]):
                # Compute Euclidean distance
                d = math.sqrt((middle_pts_cam[idx_j][0] - middle_pts_cam[idx_i][0])**2 + (middle_pts_cam[idx_j][1] - middle_pts_cam[idx_i][1])**2)
                pts_dist.append(d)

        min_label_distance = config['visual_prompting']['min_label_distance'] / zoom_ratio
        if len(pts_dist) == 0 or min(pts_dist) > min_label_distance:
            final_middle_points = middle_points
            final_contact_points_color = contact_points_color

    return final_middle_points, final_contact_points_color

def draw_label_on_image(rgb, center_coordinates, text, radius=20, thickness=2, font_scale=0.8):
    color = (0, 255, 0)
    cv2.circle(rgb, center_coordinates, radius, color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (0, 255, 0)
    line_type = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, line_type)
    text_x = center_coordinates[0] - text_width // 2
    text_y = center_coordinates[1] + text_height // 2

    cv2.putText(rgb, text, (text_x, text_y), font, font_scale, font_color, line_type)

def visualize_mask(rgb, highlight_mask):
    highlight_mask = np.stack([highlight_mask, highlight_mask, highlight_mask], axis=-1).astype(int) * 0.5
    mask_color = np.array([[255,0,0]])
    rgb = rgb * (1-highlight_mask) + mask_color * highlight_mask

    return rgb

def normalize(v):
    nv = np.array([v])
    return v / np.linalg.norm(v)

def get_orthogonal_directions(plane_normal, up_vector=[0,1,0]):
    n = normalize(plane_normal)
    v_up = normalize(up_vector)

    plane_up = v_up - np.dot(v_up, n) * n
    plane_up = normalize(plane_up)

    plane_right = np.cross(n, plane_up)
    plane_right = normalize(plane_right)

    plane_down = -plane_up
    plane_left = -plane_right

    return plane_up, plane_down, plane_right, plane_left

def draw_arrow_on_image(rgb, start_point, end_point, color=(0, 0, 255)):
    cv2.arrowedLine(rgb, start_point, end_point, color=color, thickness=4, tipLength=0.2)

def generate_visual_prompts(config):

    data_dir = os.path.join(config['data_dir'], config['object_name'])

    camera_poses = get_camera_poses()

    articulation_tree = parse_articulation_tree(os.path.join(data_dir, "gpt4o_out", "articulation_tree.txt"))

    link_hierarchy_list = parse_link_hierarchy(os.path.join(data_dir, "gpt4o_out", "link_hierarchy.txt"))

    os.makedirs(os.path.join(data_dir, "prompt_images"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "contact_points"), exist_ok=True)

    ### compute mask projection & stats for each view
    part_stat_dict = {}
    rgb_cropped_dict = {}
    highlight_mask_dict = {}
    pts_list = []
    for part in link_hierarchy_list:
        part_name = part['name']
        link_name = part_name.split("@")[0]
        part_id = part['id']
        rgb_cropped_dict[part_id] = {}
        highlight_mask_dict[part_id] = {}

        pcd_path = os.path.join(data_dir, "parts", f"{part_name}.npy")
        if os.path.exists(pcd_path) is not True:
            pts_list.append(np.zeros((0,3)))
            continue 

        xyz = np.load(pcd_path)[:, :3]
        if xyz.shape[0] == 0:
            pts_list.append(np.zeros((0,3)))
            continue

        xyz = filter_outliers_DBSCAN(xyz)
        pts_list.append(xyz)
        
        for view in range(len(camera_poses)):
            if articulation_tree[link_name]['joint_type'] not in ["fixed"]:

                rgb = cv2.imread(os.path.join(data_dir, "render", f"rgb_{view}.png"))
                H, W = 640, 640
                K = calculate_camera_intrinsics(640,640,60,60)

                part_pts_cam = project_pcd(camera_poses[view], xyz, K)

                min_x, max_x = np.min(part_pts_cam[:, 0]), np.max(part_pts_cam[:, 0])
                min_y, max_y = np.min(part_pts_cam[:, 1]), np.max(part_pts_cam[:, 1])

                x_margin = config['visual_prompting']['margin_factor'] * (max_x - min_x)
                y_margin = config['visual_prompting']['margin_factor'] * (max_y - min_y)

                min_x = max(0, int(min_x - x_margin))
                max_x = min(W, int(max_x + x_margin))
                min_y = max(0, int(min_y - y_margin))
                max_y = min(H, int(max_y + y_margin))

                box_width = max_x - min_x
                box_height = max_y - min_y

                if box_width > box_height:
                    delta = box_width - box_height
                    min_y = max(0, min_y - delta // 2)
                    max_y = min(H, max_y + delta - (delta // 2))
                    zoom_ratio = W / box_width
                else:
                    delta = box_height - box_width
                    min_x = max(0, min_x - delta // 2)
                    max_x = min(W, max_x + delta - (delta // 2))
                    zoom_ratio = H / box_height

                highres_rgb = cv2.imread(os.path.join(data_dir, "render", f"rgb_highres_{view}.png"))
                cropped_rgb = highres_rgb[int(min_y*4096/H):int(max_y*4096/H), int(min_x*4096/W):int(max_x*4096/W)]
                cropped_rgb = cv2.resize(cropped_rgb, (640,640))
                rgb_cropped_dict[part_id][view] = cropped_rgb

                highlight_mask = np.zeros((rgb.shape[0], rgb.shape[1]))
                for pix in part_pts_cam:
                    try:
                        highlight_mask[pix[1], pix[0]] = 1
                    except:
                        pass
                mask_size = np.sum(highlight_mask)
                part_stat_dict[f'part_{part_id}_{view}'] = {'mask_size': int(mask_size), 'zoom_ratio': zoom_ratio, 'min_x': min_x, 'min_y': min_y}
            
                highlight_mask_dict[part_id][view] = highlight_mask

    json.dump(part_stat_dict, open(os.path.join(data_dir, "part_stats.json"), 'w'))

    for part in link_hierarchy_list:
        part_name = part['name']
        link_name = part_name.split("@")[0]
        part_id = part['id']

        if articulation_tree[link_name]['joint_type'] in ["fixed"]:
            continue

        xyz = pts_list[part_id]
        pcd_part = o3d.geometry.PointCloud()
        pcd_part.points = o3d.utility.Vector3dVector(xyz)
        center_point_part = np.mean(xyz, axis=0).reshape(1,3)

        pts_wo_cur_part = np.concatenate([xyz_ for idx, xyz_ in enumerate(pts_list) if idx != part_id], axis=0)
        pts_wo_cur_part = filter_outliers_DBSCAN(pts_wo_cur_part, 0.01)

        pcd_rest = o3d.geometry.PointCloud()
        pcd_rest.points = o3d.utility.Vector3dVector(pts_wo_cur_part)

        ### compute contact_points
        contacts, contact_points_rest = compute_contact_points(pcd_part, pcd_rest)

        np.save(os.path.join(data_dir, "contact_points", f"contact_points_{part_id}.npy"), contact_points_rest)

        ### find the maximum number of clusters such that the labels do not overlap
        if articulation_tree[link_name]['joint_type'] in ["revolute", "continuous", "prismatic"]:

            view_list = [view_idx for view_idx in range(len(camera_poses))]
            view_list.sort(key=lambda x: part_stat_dict[f"part_{part_id}_{x}"]['mask_size'])
            top1_view = view_list[-1]

            if part_stat_dict[f"part_{part_id}_{top1_view}"]['mask_size'] < config['visual_prompting']['zoom_threshold']:
                zoom_ratio = part_stat_dict[f"part_{part_id}_{top1_view}"]['zoom_ratio']
            else:
                zoom_ratio = 1

            K = calculate_camera_intrinsics(640,640,60,60)
            label_points, contact_points_color = cluster_contact_points(contacts, camera_poses[top1_view], K, config, zoom_ratio) 

            if label_points.shape[0] > 1:
                label_points = np.concatenate([label_points, center_point_part], axis=0)

            np.save(os.path.join(data_dir, "contact_points", f"label_points_{part_id}.npy"), label_points)

            ### draw labels on images
            for view in range(len(camera_poses)):
                if articulation_tree[link_name] not in ["fixed"]:

                    rgb = cv2.imread(os.path.join(data_dir, "render", f"rgb_{view}.png"))

                    label_pts_cam = project_pcd(camera_poses[view], label_points, K)
                    contact_pts_cam = project_pcd(camera_poses[view], contacts, K)

                    label_pts_cam_zoom = copy.deepcopy(label_pts_cam)
                    label_pts_cam_zoom[:,0] = (label_pts_cam_zoom[:,0] - part_stat_dict[f'part_{part_id}_{view}']['min_x']) * part_stat_dict[f'part_{part_id}_{view}']['zoom_ratio']
                    label_pts_cam_zoom[:,1] = (label_pts_cam_zoom[:,1] - part_stat_dict[f'part_{part_id}_{view}']['min_y']) * part_stat_dict[f'part_{part_id}_{view}']['zoom_ratio']

                    rgb = cv2.imread(os.path.join(data_dir, "render", f"rgb_{view}.png"))
                    rgb_zoom = rgb_cropped_dict[part_id][view]

                    for idx, pts in enumerate(label_pts_cam):
                        draw_label_on_image(rgb, (pts[0], pts[1]), str(idx+1))
                    for idx, pts in enumerate(label_pts_cam_zoom):
                        print(pts, part_stat_dict[f'part_{part_id}_{view}']['zoom_ratio'])
                        draw_label_on_image(rgb_zoom, (pts[0], pts[1]), str(idx+1))

                    cv2.imwrite(os.path.join(data_dir, "prompt_images", f"part_{part_id}_{view}_zoom.png"), rgb_zoom)
                    cv2.imwrite(os.path.join(data_dir, "prompt_images", f"part_{part_id}_{view}.png"), rgb)
                    
                    rgb = cv2.imread(os.path.join(data_dir, "render", f"rgb_{view}.png"))
                    rgb = visualize_mask(rgb, highlight_mask_dict[part_id][view])
                    rgb[contact_pts_cam[:,1], contact_pts_cam[:,0]] = contact_points_color.astype(int)
                    for idx, pts in enumerate(label_pts_cam):
                        draw_label_on_image(rgb, (pts[0], pts[1]), str(idx+1))
                    cv2.imwrite(os.path.join(data_dir, "prompt_images", f"part_{part_id}_{view}_vis.png"), rgb)

                if articulation_tree[link_name]['joint_type'] in ["prismatic"]:

                    ### draw arrows
                    contact_points_center = np.mean(contact_points_rest, axis=0)
                    rgb = cv2.imread(os.path.join(data_dir, "render", f"rgb_{view}.png"))

                    contact_pcd = o3d.geometry.PointCloud()
                    contact_pcd.points = o3d.utility.Vector3dVector(contact_points_rest)

                    plane_model, inliers = contact_pcd.segment_plane(distance_threshold=0.002, ransac_n=3, num_iterations=1000)
                    [a, b, c, d] = plane_model

                    up, down, right, left = get_orthogonal_directions([a,b,c])
                    
                    arrow_points = np.stack([np.zeros(3),up,down,right,left], axis=0) * 0.2 + contact_points_center
                    arrow_points_2D = project_pcd(camera_poses[view], arrow_points, K)[:, :2]
                    draw_arrow_on_image(rgb, arrow_points_2D[0].tolist(), arrow_points_2D[1].tolist(), (255,0,0))
                    draw_arrow_on_image(rgb, arrow_points_2D[0].tolist(), arrow_points_2D[2].tolist(), (0,255,0))
                    draw_arrow_on_image(rgb, arrow_points_2D[0].tolist(), arrow_points_2D[3].tolist(), (0,0,255))
                    draw_arrow_on_image(rgb, arrow_points_2D[0].tolist(), arrow_points_2D[4].tolist(), (0,255,255))

                    cv2.imwrite(os.path.join(data_dir, "prompt_images", f"part_{part_id}_{view}_arrow.png"), rgb)

                    np.save(os.path.join(data_dir, "contact_points", f"arrows_{part_id}.npy"), arrow_points)

