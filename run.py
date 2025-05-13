import yaml
import os
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "PartSLIP2"))

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ArticulateAnymesh"))

from render_object import render_object
from prompts import generate_articulation_tree_known_parts
from run_partslip import partslip_segmentation
from postprocess_segmentation import postprocess_segmentation
from draw_image import generate_visual_prompts
from joint_estimation import joint_estimation
from parse_utils import articulation_tree_from_image, parse_articulation_tree

# config_path = "configs/car3.yaml"
parser = argparse.ArgumentParser(description="Script with config path argument")
parser.add_argument('--config', type=str, required=True, help='Path to the config file')

args = parser.parse_args()
config_path = args.config

with open(config_path, "r") as f:
    config = yaml.safe_load(f)
data_dir = os.path.join(config['data_dir'], config['object_name'])

if os.path.exists(os.path.join(data_dir, "render", "normal_7.png")) is not True:
    print("[ rendering object ]")
    render_object(config)

os.makedirs(os.path.join(data_dir, "gpt4o_out"), exist_ok=True)
if not os.path.exists(os.path.join(data_dir, "gpt4o_out", "articulation_tree.txt")):
    print("[ generating articulation tree ]")
    if config['parts'] != None:        
        generate_articulation_tree_known_parts(config['object_category'], config['parts']+['base'], os.path.join(data_dir, "gpt4o_out"))
    else:
        articulation_tree_from_image(config)

if os.path.exists(os.path.join(data_dir, "instance_seg")) is not True:
    print("[ performing part segmentation ]")
    if config['parts'] != None:
        part_list = config['parts']
    else:
        part_list = list(parse_articulation_tree(os.path.join(data_dir, "gpt4o_out", "articulation_tree.txt")).keys())
        part_list.remove('base')

    partslip_segmentation(config, part_list)

if os.path.exists(os.path.join(data_dir, "gpt4o_out", "link_hierarchy.txt")) is not True:
    print("[ post-processing part segmentation ]")
    postprocess_segmentation(config)

if not os.path.exists(os.path.join(data_dir, "prompt_images")) or not len(os.listdir(os.path.join(data_dir, "prompt_images"))) > 0:
    print("[ generating visual prompts ]")
    generate_visual_prompts(config)

print("[ estimating joint parameters ]")
joint_estimation(config)
