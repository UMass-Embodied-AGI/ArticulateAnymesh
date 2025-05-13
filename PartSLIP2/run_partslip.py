import os
import torch
import json
from pytorch3d.io import IO
import numpy as np
from src.utils import normalize_pc
from src.render_pc import render_pc
from src.glip_inference import dino_inference, dinox_inference, SOM_GPT_inference
from src.bbox2seg import bbox2seg
from src.gen_superpoint import gen_superpoint
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from lang_sam import LangSAM
import yaml

def partslip_segmentation(config, part_names):
    save_dir = os.path.join(config['data_dir'], config['object_name'])
    object_category = config['object_category']
    overwrite = config['overwrite'] 
    input_type = config['segmentation']['input_type']
    seg_method = config['segmentation']['seg_method']

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_type = "cuda"
        torch.cuda.set_device(device)
    else:
        device_type = "cpu"
        device = torch.device("cpu")
    
    xyz = np.load(save_dir + "/points_xyz.npy")
    rgb = np.load(save_dir + "/points_color.npy") /255
    normal = np.load(save_dir + "/points_normal.npy")
    
    with torch.autocast(device_type=device_type, enabled=False):
        pc_idx, screen_coords, num_views = render_pc(xyz, rgb, save_dir, device)
    if seg_method == "dino":
        dino = LangSAM()
        masks = dino_inference(dino, save_dir, part_names, num_views=num_views)
    elif seg_method == "dinox":
        masks = dinox_inference(save_dir, part_names, num_views=num_views)
    elif seg_method == "SoM":
        sam = sam_model_registry["vit_h"](checkpoint="./sam_vit_h_4b8939.pth").to(device)
        mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=100)
        with torch.autocast(device_type=device_type, enabled=False):
            masks = SOM_GPT_inference(mask_generator, save_dir, part_names, input_type, object_category, overwrite=overwrite, num_views=num_views, label_filter_threshold=config['segmentation']['label_filter_threshold'])

    superpoint = gen_superpoint(xyz, rgb, normal, visualize=True, save_dir=save_dir, reg=0.05)

    bbox2seg(xyz, superpoint, masks, screen_coords, pc_idx, part_names, save_dir, solve_instance_seg=True, num_view=num_views)
    