import os
import numpy as np
import cv2
import supervision as sv
from pathlib import Path
import re

# from mask_utils import _rle2mask, _string2rle

from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task

import numpy as np
from PIL import Image
from typing import List, Tuple

@staticmethod
def _string2rle(rle_str: str) -> List[int]:
    p = 0
    cnts = []
    while p < len(rle_str) and rle_str[p]:
        x = 0
        k = 0
        more = 1
        while more:
            c = ord(rle_str[p]) - 48
            x |= (c & 0x1f) << 5 * k
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << 5 * k
        if len(cnts) > 2:
            x += cnts[len(cnts) - 2]
        cnts.append(x)
    return cnts

@staticmethod
def _rle2mask(cnts: List[int], size: Tuple[int, int], label=1):
    img = np.zeros(size, dtype=np.uint8)
    ps = 0
    for i in range(0, len(cnts), 2):
        ps += cnts[i]
        for j in range(cnts[i + 1]):
            x = (ps + j) % size[1]
            y = (ps + j) // size[1]
            if y < size[0] and x < size[1]:
                img[y, x] = label
            else:
                break
        ps += cnts[i + 1]
    return img

def _rle2rgba(self, mask_obj) -> Image.Image:
    """
    Convert the compressed RLE string of mask object to png image object.
    :param mask_obj: The :class:`Mask <dds_cloudapi_sdk.tasks.ivp.IVPObjectMask>` object detected by this task
    """
    # convert rle counts to mask array
    rle = self.string2rle(mask_obj.counts)
    mask_array = self.rle2mask(rle, mask_obj.size)
    # convert the array to a 4-channel RGBA image
    mask_alpha = np.where(mask_array == 1, 255, 0).astype(np.uint8)
    mask_rgba = np.stack((255 * np.ones_like(mask_alpha),
                          255 * np.ones_like(mask_alpha),
                          255 * np.ones_like(mask_alpha),
                          mask_alpha),
                         axis=-1)
    image = Image.fromarray(mask_rgba, "RGBA")
    return image

class DINOX:
    def __init__(self):
        # token = os.getenv("DDS_CLOUDAPI_TOKEN")
        token = "10c81881124eec6d764b05ecfeab3a25"
        if not token:
            raise ValueError("API Token not found. Please set DDS_CLOUDAPI_TOKEN environment variable.")
        self.config = Config(token)
        self.client = Client(self.config)


    def get_dinox(self, image_path, input_prompts=None): 
        if image_path.startswith(('http://', 'https://')):
            infer_image_url = image_path
        else:
            print(f"Uploading image {image_path}")
            infer_image_url = self.client.upload_file(image_path)

        text_prompt = "<prompt_free>" if input_prompts is None else input_prompts

        task = V2Task(api_path="/v2/task/dinox/detection", api_body={
            "model": "DINO-X-1.0",
            "image": infer_image_url,
            "prompt": {
                "type": "text",
                "text": text_prompt,
            },
            "targets": ["bbox", "mask"],
            "bbox_threshold": 0.20,
            "iou_threshold": 0.85
        })
        task.set_request_timeout(15)
        self.client.run_task(task)
        return task.result['objects']


    def visualize_bbox_and_mask(self, predictions, img_path, output_dir, img_name):
        os.makedirs(output_dir, exist_ok=True)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            return None, None

        if not predictions:
            print(f"No objects detected in {img_path}")
            return None, None

        classes, boxes, masks, confidences = [], [], [], []

        for obj in predictions:
            if 'bbox' not in obj or 'mask' not in obj:
                print(f"No bbox or mask in {obj}, continue")
                continue

            bbox = obj['bbox']
            mask_rle = obj['mask']

            if bbox and len(bbox) == 4 and mask_rle:
                x1, y1, x2, y2 = bbox
                boxes.append([x1, y1, x2, y2])
                rle_counts = _string2rle(mask_rle['counts'])
                mask = _rle2mask(rle_counts, mask_rle['size']).astype(bool)
                masks.append(mask)

                class_name = obj.get('category')
                classes.append(class_name)
                confidences.append(obj.get('score', 1.0))

        if not boxes:
            print(f"No valid detections in {img_path}")
            return None, None

        boxes_np = np.array(boxes)
        masks_np = np.array(masks)

        unique_classes = list(set(classes))
        class_id_map = {cls: i for i, cls in enumerate(unique_classes)}
        class_ids = np.array([class_id_map[cls] for cls in classes])

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(classes, confidences)
        ]

        detections = sv.Detections(
            xyxy=boxes_np,
            mask=masks_np,
            class_id=class_ids,
        )

        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(output_dir, f"{img_name}_bbox_mask.jpg"), annotated_frame)

        print(f"visualized DINOX results to {output_dir}/{img_name}_bbox_mask.jpg")

        return boxes, masks