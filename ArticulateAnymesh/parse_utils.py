import os
import re

from prompts import generate_articulation_tree, recognize_parts_from_image

def extract_output_format(txt_file, format_name):
    with open(txt_file, 'r') as file:
        response = file.read()
    pattern = rf'```{format_name}\s+(.*?)```'
    form = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

    return form

def articulation_tree_from_image(config):
    data_dir = os.path.join(config['data_dir'], config['object_name'])
    save_file = os.path.join(data_dir, "gpt4o_out", "part_recognition.txt")
    img_paths = [os.path.join(data_dir, "render", f"rgb_{j}.png") for j in range(8)]
    if os.path.exists(save_file) is not True:
        recognize_parts_from_image(img_paths, config['object_category'], save_file)

    with open(save_file, 'r') as file:
        part_list_txt = file.read()
    pattern = r'```part_list\s+(.*?)```'
    part_list_txt = re.findall(pattern, part_list_txt, re.DOTALL | re.IGNORECASE)[0]

    part_list = []
    for line in part_list_txt.split("\n"):
        if len(line) > 0:
            part_name = line.split("part_name:")[-1].split(";")[0].strip()
            part_list.append(part_name)

    generate_articulation_tree(config['object_category'], part_list, os.path.join(data_dir, "gpt4o_out"))


def parse_articulation_tree(articulation_tree_path):

    with open(articulation_tree_path, 'r') as file:
        articulation_tree = file.read()

    sections = articulation_tree.split('joints:')
    links_section = sections[0].split('links:')[1].strip()
    joints_section = sections[1].strip()

    info_dict = {}
    info_dict['base'] = {'joint_type': 'fixed', 'parent':None}
    for line in links_section.split('\n'):
        line = line.strip()
        if len(line) > 0 and line[0] == "(":
            link_name = line.split("link_name:")[-1].split(";")[0].strip().lower()
            info_dict[link_name] = {'joint_type': 'fixed'}

    for line in joints_section.split('\n'):
        line = line.strip()
        if line[0] == "(":
            parent_link = line.split("parent_link:")[-1].split(";")[0].strip().lower()
            child_link = line.split("child_link:")[-1].split(";")[0].strip().lower()
            joint_type = line.split("joint_type:")[-1].split(";")[0].strip().lower()
            info_dict[child_link]['joint_type'] = joint_type
            info_dict[child_link]['parent'] = parent_link
            info_dict[parent_link]['child'] = child_link

    return info_dict

def parse_link_hierarchy(link_hierarchy_path):

    with open(link_hierarchy_path, 'r') as file:
        link_hierarchy = file.read()
        
    link_hierarchy_list = []
    pattern = r'```link_names\s+(.*?)```'
    for line in re.findall(pattern, link_hierarchy, re.DOTALL | re.IGNORECASE)[0].split("\n"):
        line = line.strip()
        if len(line) > 0:
            link_name = line.split(".")[1].strip()
            link_id = line.split(".")[0].strip()
            link_hierarchy_list.append({'id': int(link_id), 'name': link_name})

    pattern = r'```parent_links\s+(.*?)```'
    for line in re.findall(pattern, link_hierarchy, re.DOTALL | re.IGNORECASE)[0].split("\n"):
        line = line.strip()
        if len(line) > 0:
            parent_name = line.split(".")[1].strip()
            link_id = line.split(".")[0].strip()
            link_hierarchy_list[int(link_id)]['parent'] = parent_name

    return link_hierarchy_list

def parse_joint_classification(articulation_tree, link_hierarchy):
    
    for part in link_hierarchy_list:
        part_name = part['name']
        link_name = part_name.split("@")[0]
        joint_type = articulation_tree[link_name]

        if joint_type in ["revolute", "continuous"] and os.path.exists(os.path.join(data_dir, "gpt4o_out", "revolute_classification", f"{link_name}.txt")) is not True:
            print("selecting method to obtain the joints for each revolute joint")
            classify_revolute_joint([], object_name, link_name, os.path.join(data_dir, "gpt4o_out", "revolute_classification"))

        if joint_type == "prismatic" and os.path.exists(os.path.join(data_dir, "gpt4o_out", "prismatic_classification", f"{link_name}.txt")) is not True:
            print("selecting method to obtain the joints for each prismatic joint")
            classify_prismatic_joint([], object_name, link_name, os.path.join(data_dir, "gpt4o_out", "prismatic_classification"))
