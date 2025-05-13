import os
import torch
import numpy as np

def normalize_pc(pc_file, save_dir, io, device, save_normalized_pc=False, do_normalize=True):
    pc = io.load_pointcloud(pc_file, device = device)
    xyz = pc.points_padded().reshape(-1,3)
    rgb = pc.features_padded().reshape(-1,3)
    if do_normalize:
        xyz = xyz - xyz.mean(axis=0)
        xyz = xyz / torch.norm(xyz, dim=1, p=2).max().item()
    xyz = xyz.cpu().numpy()
    rgb = rgb.cpu().numpy()
    if save_normalized_pc:
        save_colored_pc(os.path.join(save_dir, "normalized_pc.ply"), xyz, rgb)
    return xyz, rgb

def save_colored_pc(file_name, xyz, rgb):
    n = xyz.shape[0]
    if rgb.max() < 1.1:
        rgb = (rgb * 255).astype(np.uint8)
    f = open(file_name, "w")
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write("element vertex %d\n" % n)
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")
                
    for i in range(n):
        f.write("%f %f %f %d %d %d\n" % (xyz[i][0], xyz[i][1], xyz[i][2], rgb[i][0], rgb[i][1], rgb[i][2]))


# def get_covered(bb1, bb2):
#     # Coordinates of intersection
#     x_left = max(bb1['x1'], bb2['x1'])
#     y_top = max(bb1['y1'], bb2['y1'])
#     x_right = min(bb1['x2'], bb2['x2'])
#     y_bottom = min(bb1['y2'], bb2['y2'])

#     # Compute width and height of intersection
#     inter_width = max(0, x_right - x_left)
#     inter_height = max(0, y_bottom - y_top)
#     intersection_area = inter_width * inter_height

#     # Compute area of bb1
#     bb1_area = max(0, bb1['x2'] - bb1['x1']) * max(0, bb1['y2'] - bb1['y1'])

#     # Avoid division by zero
#     if bb1_area == 0:
#         return 0.0

#     return intersection_area / bb1_area



def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2'] + 1e-6
    assert bb1['y1'] < bb1['y2'] + 1e-6
    assert bb2['x1'] < bb2['x2'] + 1e-6
    assert bb2['y1'] < bb2['y2'] + 1e-6

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_cover(bb1, bb2):
    """
    Calculate the portion of bb1 covered by bb2.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2'] + 1e-6
    assert bb1['y1'] < bb1['y2'] + 1e-6
    assert bb2['x1'] < bb2['x2'] + 1e-6
    assert bb2['y1'] < bb2['y2'] + 1e-6

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    cover = intersection_area / float(bb1_area)
    assert cover >= 0.0
    assert cover <= 1.0
    return cover