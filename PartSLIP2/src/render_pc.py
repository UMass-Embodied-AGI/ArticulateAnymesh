from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    NormWeightedCompositor
)
from pytorch3d.structures import Pointclouds
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.utils import save_colored_pc

def render_single_view(pc, view, device, background_color=(1,1,1), resolution=800, camera_distance=2.2, point_size=0.005, points_per_pixel=1, bin_size=0, znear=0.01):
    R, T = look_at_view_transform(camera_distance, view[0], view[1])
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=znear, fov=60)

    raster_settings = PointsRasterizationSettings(
        image_size=resolution, 
        radius=point_size,
        points_per_pixel=points_per_pixel,
        bin_size=bin_size,
    )
    compositor=NormWeightedCompositor(background_color=background_color)
    
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=compositor
    ).to(device)

    img = renderer(pc)
    pc_idx = rasterizer(pc).idx

    raster_settings = PointsRasterizationSettings(
        image_size=resolution, 
        radius=point_size,
        points_per_pixel=150,
        bin_size=bin_size,
    )
    
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=compositor
    ).to(device)
    rast_res = rasterizer(pc)
    pc_idx = rast_res.idx
    pc_z = rast_res.zbuf
    pc_z_diff = pc_z - pc_z[:,:,:,:1]
    pc_idx[pc_z_diff >= 0.01] = -1
    screen_coords = cameras.transform_points_screen(pc._points_list[0], image_size=(resolution, resolution)).to(pc.points_packed().dtype)
    return img, pc_idx, screen_coords
    
def render_pc(xyz, rgb, save_dir, device, view_set=0):
    pc = Pointclouds(points=[torch.Tensor(xyz).to(device)], features=[torch.Tensor(rgb).to(device)])

    img_dir = os.path.join(save_dir, "rendered_img")
    os.makedirs(img_dir, exist_ok=True)

    if view_set == 0:
        views = [[15,0], [15,45], [15,90], [15,135], [15,180], [15,225], [15,270], [15,315]]
    elif view_set == 1:
        views = [[15,0], [15,45], [15,90], [15,135], [15,180], [15,225], [15,270], [15,315],   [-20,0+22.5], [-20,45+22.5], [-20,90+22.5], [-20,135+22.5], [-20,180+22.5], [-20,225+22.5], [-20,270+22.5], [-20,315+22.5],   [35,0+22.5], [35,45+22.5], [35,90+22.5], [35,135+22.5], [35,180+22.5], [35,225+22.5], [35,270+22.5], [35,315+22.5]]
    pc_idx_list = []
    screen_coords_list = []

    for i, view in enumerate(views):
        img, pc_idx, screen_coords = render_single_view(pc, view, device, camera_distance=2.0, resolution=160, point_size=0.02) # low resolution to filter out far points
        screen_coords = screen_coords*4
        img = torch.nn.functional.interpolate(torch.tensor(img.permute(0,3,1,2), dtype=float), size=(640,640)).to(int).permute(0,2,3,1)
        plt.imsave(os.path.join(img_dir, f"{i}.png"), img[0, ..., :3].cpu().numpy() * 0.99999)
        pc_idx_full = torch.nn.functional.interpolate(torch.tensor(pc_idx.permute(0,3,1,2), dtype=float), size=(640,640)).permute(0,2,3,1).to(int)
        pc_idx_list.append(pc_idx_full)
        screen_coords_list.append(screen_coords)

    pc_idx = torch.cat(pc_idx_list, dim=0).squeeze()
    screen_coords = torch.cat(screen_coords_list, dim=0).reshape(len(views),-1, 3)[...,:2]

    np.save(f"{save_dir}/idx.npy", pc_idx.cpu().numpy())
    np.save(f"{save_dir}/coor.npy", screen_coords.cpu().numpy())
    return pc_idx.cpu().numpy(), screen_coords.cpu().numpy(), len(views)