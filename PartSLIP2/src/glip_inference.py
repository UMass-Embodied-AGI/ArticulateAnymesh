import os
from PIL import Image
import pickle
import numpy as np
import json
import matplotlib.pyplot as plt

# from maskrcnn_benchmark.config import cfg
# from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

import cv2
import re
import supervision as sv
from lang_sam import LangSAM
from lang_sam.utils import draw_image

from scipy.ndimage import generic_filter
from collections import Counter

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from src.segmentation_prompts import *
from src.dinox_detector import DINOX, _string2rle, _rle2mask

def load_img(file_name):
    pil_image = Image.open(file_name).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def draw_rectangle(img, x0, y0, x1, y1):
    color = np.random.rand(3) * 255
    img = img.astype(np.float64)
    img[y0:y1, x0-1:x0+2, :3] = color
    img[y0:y1, x1-1:x1+2, :3] = color
    img[y0-1:y0+2, x0:x1, :3] = color
    img[y1-1:y1+2, x0:x1, :3] = color
    img[y0:y1, x0:x1, :3] /= 2
    img[y0:y1, x0:x1, :3] += color * 0.5
    img = img.astype(np.uint8)
    return img

def dino_inference(dino, save_dir, part_names, num_views=10):
    pred_dir = os.path.join(save_dir, "dino_pred")
    os.makedirs(pred_dir, exist_ok = True)
    seg_masks = [[] for _ in range(num_views)]
    preds = [[] for _ in range(num_views)]
    for i in range(num_views):
        image_pil = Image.open(os.path.join(save_dir, "render", f"rgb_{i}.png")).convert("RGB")
        text_prompt = ".".join(part_names)
        results = dino.predict([image_pil], [text_prompt])[0]

        if len(results["masks"]) != 0:
            output_image = draw_image(
                image_pil,
                results["masks"],
                results["boxes"],
                results["scores"],
                results["labels"],
            )
            output_image = Image.fromarray(np.uint8(output_image)).convert("RGB")
            output_image.save(f'{pred_dir}/{i}.png')

            for j in range(len(results["masks"])):
                try:
                    seg_masks[i].append((results["masks"][j]==1, part_names.index(results["labels"][j]), results["boxes"][j]))
                except:
                    for label in results["labels"][j].split(" "):
                        seg_masks[i].append((results["masks"][j]==1, part_names.index(label), results["boxes"][j]))

    return seg_masks
                
def dinox_inference(save_dir, part_names, num_views=10):

    seg_masks = [[] for _ in range(num_views)]
    os.makedirs(os.path.join(save_dir, "dinox_pred"), exist_ok=True)

    # dds cloudapi for DINO-X
    from dds_cloudapi_sdk import Config
    from dds_cloudapi_sdk import Client

    # API_TOKEN = "10c81881124eec6d764b05ecfeab3a25"
    API_TOKEN = os.environ['DINOX_API_TOKEN']
    config = Config(API_TOKEN)

    # Step 2: initialize the client
    client = Client(config)

    # Step 3: Run DINO-X task
    # if you are processing local image file, upload them to DDS server to get the image url
    for i in range(num_views):
        img_path = os.path.join(save_dir, "render", f"rgb_{i}.png")
        text_prompt = " . ".join(part_names)
        image_url = client.upload_file(img_path)

        if os.path.exists(os.path.join(save_dir, "dinox_pred", f'{i}.pickle')):

            with open(os.path.join(save_dir, "dinox_pred", f'{i}.pickle'), 'rb') as file:
                predictions = pickle.load(file)

        else:

            dinox = DINOX()

            predictions = dinox.get_dinox(image_url, text_prompt)

            with open(os.path.join(save_dir, "dinox_pred", f'{i}.pickle'), "wb") as file:
                pickle.dump(predictions, file)

        classes = [x.strip().lower() for x in text_prompt.split('.') if x]
        class_name_to_id = {name: id for id, name in enumerate(classes)}
        class_id_to_name = {id: name for name, id in class_name_to_id.items()}

        boxes = []
        masks = []
        confidences = []
        class_names = []
        class_ids = []

        for idx, obj in enumerate(predictions):
            if obj['score'] < 0.3:
                continue
            boxes.append(obj['bbox'])
            masks.append(_rle2mask(_string2rle(obj['mask']['counts']), obj['mask']['size']))  # convert mask to np.array using DDS API
            confidences.append(obj['score'])
            cls_name = obj['category'].lower().strip()
            class_names.append(cls_name)
            class_ids.append(class_name_to_id[cls_name])

        boxes = np.array(boxes)
        masks = np.array(masks)
        class_ids = np.array(class_ids)
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]
        # for visualization

        img = cv2.imread(img_path)
        detections = sv.Detections(
            xyxy = boxes,
            mask = masks.astype(bool),
            class_id = class_ids,
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(save_dir, "dinox_pred", f"{i}.png"), annotated_frame)

        
        for j in range(masks.shape[0]):

            seg_masks[i].append((masks[j]==1, part_names.index(class_names[j]), boxes[j]))

    return seg_masks

def most_frequent_label(kernel):
    labels, counts = np.unique(kernel, return_counts=True)
    return labels[np.argmax(counts)]

def smooth_segmentation_map(input_map, kernel_size=3):
    return generic_filter(input_map, most_frequent_label, size=kernel_size, mode='reflect')

def SOM_GPT_inference(sam_predictor, save_dir, part_names, input_type, object_name, num_views=8, overwrite=False, label_filter_threshold=1000):

    ### generate SOM images
    seg_masks = [[] for _ in range(num_views)]

    for i in range(num_views):
        img = f"{i}.png"
        rgb_path = os.path.join(save_dir, "render", f"rgb_{i}.png")
        normal_path = os.path.join(save_dir, "render", f"normal_{i}.png")
        if input_type == 'rgb':
            image = cv2.imread(rgb_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = sam_predictor.generate(image)
        elif input_type == 'normal':
            image = cv2.imread(normal_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = sam_predictor.generate(image)
        elif input_type == 'both':
            image = cv2.imread(rgb_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = sam_predictor.generate(image)

            image = cv2.imread(normal_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = masks + sam_predictor.generate(image)

    
        os.makedirs(os.path.join(save_dir, "SoM"), exist_ok=True)
        mask_in_one = -np.ones_like(image)[:,:,0]
        rgb_in_one = np.zeros_like(image)
        masks = [m for m in masks if m['area'] >= label_filter_threshold]
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        for idx, mask in enumerate(sorted_masks):
            mask_in_one[mask['segmentation']] = idx + 1
            rgb_in_one[mask['segmentation']] = np.random.random(3) * 255

        cv2.imwrite(os.path.join(save_dir, "SoM", img), rgb_in_one)
        mask_in_one_smoothed = smooth_segmentation_map(mask_in_one, 5)

        ### break disconnected mask into smaller mask
        mask_ids = np.unique(mask_in_one_smoothed)
        max_id = mask_ids[-2] if 255 in mask_ids.tolist() else mask_ids[-1]
        for mask_id in mask_ids:
            mask_w_id = mask_in_one_smoothed == mask_id
            num_labels, labels_im = cv2.connectedComponents(mask_w_id.astype(np.uint8))
            if num_labels > 2:
                for cc in np.unique(labels_im):
                    if cc == 0 or cc == 1:
                        continue
                    else:
                        mask_in_one_smoothed[labels_im==cc] = max_id + 1
                        max_id += 1

        ### delete small masks and background masks
        background_mask = cv2.imread(os.path.join(save_dir, "render", f"mask_{i}.png"))
        background_mask = (background_mask < 100)[:,:,0]
        mask_ids = np.unique(mask_in_one_smoothed)
        new_mask_id = 1
        for mask_id in mask_ids:
            if mask_id == 255:
                continue
            mask = (mask_in_one_smoothed == mask_id).astype(np.uint8)
            if np.sum(mask) < label_filter_threshold:
                mask_in_one_smoothed[mask_in_one_smoothed == mask_id] = 255
            elif np.sum(np.logical_and(mask, background_mask)) / np.sum(mask) >= 0.7:
                mask_in_one_smoothed[mask_in_one_smoothed == mask_id] = 255
            else:
                mask_in_one_smoothed[mask_in_one_smoothed == mask_id] = new_mask_id
                new_mask_id += 1

        ### draw the final seg image
        rgb_smoothed_mask = np.zeros_like(image)
        for mask_id in np.unique(mask_in_one_smoothed).tolist():
            rgb_smoothed_mask[mask_in_one_smoothed[:,:] == mask_id] = np.random.random(3) * 255
        cv2.imwrite(os.path.join(save_dir, "SoM", img.replace(".png", f"_final.png")), np.flip(rgb_smoothed_mask, axis=-1))

        np.save(os.path.join(save_dir, "SoM", img.replace(".png", f"_final.npy")), mask_in_one_smoothed)

        ### draw mask for the whole image
        from visualizer import Visualizer
        image = cv2.imread(rgb_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        visual = Visualizer(image)
        
        for mask_id in np.unique(mask_in_one_smoothed).tolist():
            mask = (mask_in_one_smoothed == mask_id).astype(np.uint8)
            if np.sum(mask) < label_filter_threshold:
                continue
            if mask_id == 255:
                continue
            color_mask = np.random.random((1, 3)).tolist()[0]
            demo = visual.draw_binary_mask_with_number(mask, text=str(mask_id), label_mode='1', alpha=0.1, anno_mode=['Mask', 'Mark'])

        im = demo.get_image()
        cv2.imwrite(os.path.join(save_dir, "SoM", img.replace(".png", f"_labeled.png")), np.flip(im, axis=-1))

    ### prompt GPT to select the parts
    txt_path = os.path.join(save_dir, "gpt4o_out", "multiview_recognition.txt")
    if overwrite or os.path.exists(txt_path) is not True:
        os.makedirs(os.path.join(save_dir, "gpt4o_out"), exist_ok=True)
        img_paths = ["%s/render/rgb_%d.png" % (save_dir, j) for j in range(num_views)]
        multiview_parts_recognition(img_paths, object_name, num_views, part_names, os.path.join(save_dir, "gpt4o_out"))
    
    with open(txt_path, "r") as file:
        response = file.read()
        
    pattern = r'```part_recognition\s+(.*?)```'

    ve = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)[0]
    part_img_dict = {}
    for line in ve.split('\n'):
        if len(line.strip()) == 0:
            continue
        part_name = line.split(';')[0].replace("part_name:", "").strip()
        view_indices = [int(x.strip()) for x in line.split(';')[1].replace("views:", "").split(",")]
        part_img_dict[part_name] = view_indices

    for img_idx in range(num_views):
        if not overwrite and os.path.exists(os.path.join(save_dir, "SoM", str(img_idx), "masks_to_links.txt")):
            continue
        labeled_img_path = os.path.join(save_dir, "SoM", f"{img_idx}_labeled.png")
        part_list_img = [part for part in part_names if img_idx in part_img_dict[part]]
        if len(part_list_img) == 0:
            continue
        os.makedirs(os.path.join(save_dir, "SoM", str(img_idx)), exist_ok=True)
        merge_parts([labeled_img_path], object_name, part_list_img, os.path.join(save_dir, "SoM", str(img_idx)))

    semantics_dict = {}
    for img_idx in range(num_views):
        semantics_file = os.path.join(save_dir, "SoM", str(img_idx), "masks_to_links.txt")
        if os.path.exists(semantics_file) is not True:
            continue

        with open(semantics_file, 'r') as file:
            semantics = file.read()

        lines = semantics.split("\n")
        semantics_dict[img_idx] = {}
        for line in lines:
            label_txt = "labels:"
            name_txt = "name:"
                
            if label_txt in line:
                link_name = line.split(name_txt)[-1].split(";")[0].strip()
                labels = line.split(label_txt)[-1].split("\n")[0].strip()
                semantics_dict[img_idx][link_name] = []
                for label in labels.split(","):
                    try:
                        int_label = int(label.strip())
                    except:
                        continue
                    semantics_dict[img_idx][link_name].append(int_label)

        mask_in_one = np.load(os.path.join(save_dir, "SoM", f"{img_idx}_final.npy"))

        ### gather the results
        for part_idx, part in enumerate(part_names): 
            if img_idx in part_img_dict[part]:
                for unique_part in semantics_dict[img_idx].keys():
                    if part in unique_part:
                        part_mask = np.zeros_like(mask_in_one)
                        for segment_idx in semantics_dict[img_idx][unique_part]:
                            segment_mask = mask_in_one == segment_idx
                            part_mask = np.logical_or(part_mask, segment_mask)

                        if np.sum(part_mask) == 0:
                            continue

                        r, c = np.where(part_mask)
                        part_bbox = np.array([np.min(c), np.min(r), np.max(c), np.max(r)])

                        seg_masks[img_idx].append((part_mask, part_idx, part_bbox))

    return seg_masks
    