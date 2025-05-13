import os
import cv2
import json
import open3d as o3d
import numpy as np
import trimesh

from parse_utils import extract_output_format
from prompts import *
from parse_utils import parse_articulation_tree, parse_link_hierarchy, parse_joint_classification

def choose_prompting_method(config, articulation_tree, link_hierarchy_list):

    data_dir = os.path.join(config['data_dir'], config['object_name'])
    for part in link_hierarchy_list:
        part_name = part['name']
        link_name = part_name.split("@")[0]
        joint_type = articulation_tree[link_name]['joint_type']

        if joint_type in ["revolute", "continuous"]:
            txt_file = os.path.join(data_dir, "gpt4o_out", "revolute_method", f"{link_name}.txt")
            os.makedirs(os.path.join(data_dir, "gpt4o_out", "revolute_method"), exist_ok=True)
            if os.path.exists(txt_file) is not True:
                select_revolute_method([], config['object_category'], link_name, txt_file)

            hinge_info = extract_output_format(txt_file, 'hinge_info')[0]
            method_idx = int(hinge_info.split("choice: ")[1].replace("(","").replace(")","").split("\n")[0])
            articulation_tree[link_name]['method'] = method_idx

        if joint_type == "prismatic":
            txt_file = os.path.join(data_dir, "gpt4o_out", "prismatic_method", f"{link_name}.txt")
            os.makedirs(os.path.join(data_dir, "gpt4o_out", "prismatic_method"), exist_ok=True)
            if os.path.exists(txt_file) is not True:
                select_prismatic_method([], config['object_category'], link_name, txt_file)

            slider_info = extract_output_format(txt_file, 'slider_info')[0]
            method = slider_info.split("choice: ")[1].replace("(","").replace(")","").split("\n")[0]
            if "inward" in method or "outward" in method:
                method_idx = 1
            elif "surface" in method:
                method_idx = 2
            articulation_tree[link_name]['method'] = method_idx

    return articulation_tree


def get_mobility_json(articulation_tree, link_hierarchy_list):

    mobility_dict = []
    for part in link_hierarchy_list:
        part_name = part['name']
        link_name = part_name.split("@")[0]
        part_id = part['id']
        parent_name = part['parent'] if link_name != 'base' else None
        joint_type = articulation_tree[link_name]['joint_type']
        
        if joint_type in ["revolute", "continuous"]:
            joint_data = {"axis": {"origin": part['origin'], "direction": part['direction']}}
        elif joint_type in ["prismatic"]:
            joint_data = {"axis": {"origin": [0,0,0], "direction": part['direction']}}

        elif joint_type in ["fixed"]:
            joint_data = {}

        mobility_dict.append({"id": part_id, "parent": parent_name, "joint": joint_type, "name": part_name, "jointData": joint_data})

    return mobility_dict

def joint_estimation(config):

    data_dir = os.path.join(config['data_dir'], config['object_name'])
    object_category = config['object_category']

    articulation_tree = parse_articulation_tree(os.path.join(data_dir, "gpt4o_out", "articulation_tree.txt"))

    link_hierarchy_list = parse_link_hierarchy(os.path.join(data_dir, "gpt4o_out", "link_hierarchy.txt"))

    articulation_tree = choose_prompting_method(config, articulation_tree, link_hierarchy_list)

    parts_stats = json.load(open(os.path.join(data_dir, "part_stats.json")))
    for part in link_hierarchy_list:
        part_name = part['name']
        part_id = part['id']
        link_name = part_name.split("@")[0]
        joint_type = articulation_tree[link_name]['joint_type']

        if joint_type in ["fixed"]:
            continue
        elif joint_type in ["prismatic"]:

            method_idx = articulation_tree[link_name]["method"]
            
            if method_idx == 1:
                # perpendicular to the connecting area
                contact_points = np.load(os.path.join(data_dir, "contact_points", f"contact_points_{part_id}.npy"))
                label_points = np.load(os.path.join(data_dir, "contact_points", f"label_points_{part_id}.npy"))
                contact_pcd = o3d.geometry.PointCloud()
                contact_pcd.points = o3d.utility.Vector3dVector(contact_points)
                plane_model, inliers = contact_pcd.segment_plane(distance_threshold=0.002, ransac_n=3, num_iterations=1000)
                [a, b, c, d] = plane_model
                link_hierarchy_list[part_id]['direction'] = [a,b,c]
                
            elif method_idx == 2:
                # arrow
                view_list = [view for view in range(len(parts_stats)//(len(link_hierarchy_list)))]
                view_list.sort(key=lambda x: parts_stats[f"part_{part_id}_{x}"]['mask_size'])
                view = view_list[-1]
                save_path = os.path.join(data_dir, "gpt4o_out", "arrow_selection")
                os.makedirs(save_path, exist_ok=True)
                prompt_img_path = os.path.join(data_dir, "prompt_images", f"part_{part_id}_{view}_arrow.png")

                txt_file = os.path.join(save_path, f"arrow_selection_{part_id}_{view}.txt")
                if os.path.exists(txt_file) is not True:
                    select_translation_direction([prompt_img_path], object_category, part_name, txt_file)

                with open(txt_file, 'r') as file:
                    arrow_selection_txt = file.read()
                selections = arrow_selection_txt.split("selected arrow:")[-1].split("\n")[0].split(",")
                selections = [s.strip() for s in selections]
                arrow_points = np.load(os.path.join(data_dir, "contact_points", f"arrows_{part_id}.npy"))

                if len(selections) == 1:
                    
                    if "blue" == selections[0]:
                        link_hierarchy_list[part_id]['direction'] = arrow_points[1] - arrow_points[0]
                    elif "green" == selections[0]:
                        link_hierarchy_list[part_id]['direction'] = arrow_points[2] - arrow_points[0]
                    elif "red" == selections[0]:
                        link_hierarchy_list[part_id]['direction'] = arrow_points[3] - arrow_points[0]
                    elif "yellow" == selections[0]:
                        link_hierarchy_list[part_id]['direction'] = arrow_points[4] - arrow_points[0]
                    link_hierarchy_list[part_id]['direction'] = link_hierarchy_list[part_id]['direction'].tolist()

                elif len(selections) == 2:

                    if "blue" in selections and "green" in selections:
                        link_hierarchy_list[part_id]['direction'] = arrow_points[1] - arrow_points[0]
                    elif "red" in selections and "yellow" in selections:
                        link_hierarchy_list[part_id]['direction'] = arrow_points[3] - arrow_points[0]
                    link_hierarchy_list[part_id]['direction'] = link_hierarchy_list[part_id]['direction'].tolist()

            else:
                raise ValueError("Invalid method")

            
        elif joint_type in ["revolute", "continuous"]:

            view_list = [view for view in range(len(parts_stats)//(len(link_hierarchy_list)))]
            view_list.sort(key=lambda x: parts_stats[f"part_{part_id}_{x}"]['mask_size'])
            view = view_list[-1]
            save_path = os.path.join(data_dir, "gpt4o_out", "hinge_selection")
            os.makedirs(save_path, exist_ok=True)
            if parts_stats[f"part_{part_id}_{view}"]['mask_size'] < config["visual_prompting"]["zoom_threshold"]:
                prompt_img_path = os.path.join(data_dir, "prompt_images", f"part_{part_id}_{view}_zoom.png")
            else:
                prompt_img_path = os.path.join(data_dir, "prompt_images", f"part_{part_id}_{view}.png")
            method_idx = articulation_tree[link_name]['method']
            label_points = np.load(os.path.join(data_dir, "contact_points", f"label_points_{part_id}.npy"))
            num_label_points = label_points.shape[0]
            if method_idx == 1:
                txt_file = os.path.join(save_path, f"hinge_selection_{part_id}_{view}.txt")
                if num_label_points <= 1:
                    raise ValueError("Not enough label points")
                else:
                    if os.path.exists(txt_file) is not True:
                        select_correct_hinge_two_points([prompt_img_path], object_category, part_name, txt_file)
                
                with open(txt_file, 'r') as file:
                    hinge_selection_txt = file.read()
                selections = hinge_selection_txt.split("selected ids:")[-1].split("\n")[0].split(",")

                points = []
                for label_idx in selections:
                    if int(label_idx) > 0:
                        points.append(label_points[int(label_idx.strip())-1]) # label starts from 1
                points = np.stack(points, axis=0)

                mean = points.mean(axis=0)
                points_centered = points - mean
                U, S, Vt = np.linalg.svd(points_centered)

                link_hierarchy_list[part_id]['direction'] = Vt[0].tolist()
                link_hierarchy_list[part_id]['origin'] = mean.tolist()
                
            elif method_idx == 2:
                txt_file = os.path.join(save_path, f"hinge_selection_{part_id}_{view}.txt")
                if num_label_points == 0:
                    raise ValueError("Not enough label points")
                else:
                    if os.path.exists(txt_file) is not True:
                        select_correct_hinge_one_point([prompt_img_path], object_category, part_name, txt_file)

                with open(txt_file, 'r') as file:
                    hinge_selection_txt = file.read()
                selections = hinge_selection_txt.split("selected ids:")[-1].split("\n")[0].split(",")
                contact_points = np.load(os.path.join(data_dir, "contact_points", f"contact_points_{part_id}.npy"))
                points = []
                for label_idx in selections:
                    points.append(label_points[int(label_idx.strip())-1])
                points = np.stack(points, axis=0)
                assert len(points) == 1
                
                contact_pcd = o3d.geometry.PointCloud()
                contact_pcd.points = o3d.utility.Vector3dVector(contact_points)
                plane_model, inliers = contact_pcd.segment_plane(distance_threshold=0.002, ransac_n=3, num_iterations=1000)
                [a, b, c, d] = plane_model
                link_hierarchy_list[part_id]['direction'] = [a, b, c]
                link_hierarchy_list[part_id]['origin'] = points[0].tolist()

            else:
                raise ValueError("Invalid method")

        else:
            continue

        ### save joint for visualization
        cylinder = trimesh.creation.cylinder(radius=0.01, height=5)
        align_matrix = trimesh.geometry.align_vectors([0, 0, 1], np.array(link_hierarchy_list[part_id]['direction']))
        cylinder.apply_transform(align_matrix)
        origin = link_hierarchy_list[part_id]['origin'] if 'origin' in link_hierarchy_list[part_id] else label_points[-1]
        cylinder.apply_translation(origin)
        os.makedirs(os.path.join(data_dir, "joint_visualization"), exist_ok=True)
        cylinder.export(os.path.join(data_dir, "joint_visualization", f"joint_{part_name}.obj"))

    mobility_json = get_mobility_json(articulation_tree, link_hierarchy_list)
    json.dump(mobility_json, open(os.path.join(data_dir, "mobility.json"), 'w'))