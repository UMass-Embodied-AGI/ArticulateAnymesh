
import os
import cv2
import trimesh
import yaml
import numpy as np

import genesis as gs

from scipy.spatial.transform import Rotation as R


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


def get_camera_poses(use_24_cams=False):
    
    H, W = 640, 640

    camera_poses = {}

    if use_24_cams:
        elev_angles = [-15, 20, -35]
        yaw_angles = [np.radians(45 * i) for i in range(8)] + [np.radians(22.5 + 45 * i) for i in range(8)] + [np.radians(22.5 + 45 * i) for i in range(8)]
    else:
        elev_angles = [-15]
        yaw_angles = [np.radians(45 * i) for i in range(8)]

    for j, elev_angle in enumerate(elev_angles):
        for i in range(8):
            cam_pose = np.array([
                [1.0,  0.0,  0.0,  0.0],
                [0.0,  1.0,  0.0,  0.0],
                [0.0,  0.0,  1.0,  2.0],
                [0.0,  0.0,  0.0,  1.0]
            ])

            elev_angle_ = np.radians(elev_angle)
            cos_angle = np.cos(elev_angle_)
            sin_angle = np.sin(elev_angle_)

            R_x_45 = np.array([
                [1.0,       0.0,       0.0,        0.0],
                [0.0,       cos_angle, -sin_angle, 0.0],
                [0.0,       sin_angle, cos_angle,  0.0],
                [0.0,       0.0,       0.0,        1.0]
            ])

            cam_pose = np.dot(R_x_45, cam_pose)

            yaw_angle = yaw_angles[j*8+i]
            cos_angle = np.cos(yaw_angle)
            sin_angle = np.sin(yaw_angle)

            R_y_45 = np.array([
                [cos_angle, 0.0, sin_angle, 0.0],
                [0.0,       1.0, 0.0,       0.0],
                [-sin_angle,0.0, cos_angle, 0.0],
                [0.0,       0.0, 0.0,       1.0]
            ])
            cam_pose = np.dot(R_y_45, cam_pose)
            camera_poses[j*8+i] = np.linalg.inv(cam_pose @ np.diag([1,-1,-1,1]))

    return camera_poses

def render_object(config):

    H, W = 640, 640

    data_dir = os.path.join(config['data_dir'], config['object_name'])
    camera_poses_render = get_camera_poses()

    if os.path.exists(os.path.join(data_dir, "mesh.ply")):
        mesh_path = os.path.join(data_dir, "mesh.ply")
    elif os.path.exists(os.path.join(data_dir, "mesh.obj")):
        mesh_path = os.path.join(data_dir, "mesh.obj")
    elif os.path.exists(os.path.join(data_dir, "mesh.glb")):
        mesh_path = os.path.join(data_dir, "mesh.glb")
    else:
        raise Exception("mesh file non-existent")
        
    mesh = trimesh.load(mesh_path, force='mesh')

    ### center the loaded mesh
    center = np.sum(mesh.bounding_box.bounds, axis=0) / 2
    translation_matrix = trimesh.transformations.translation_matrix(-center)
    radius = np.linalg.norm(mesh.bounding_box.extents)
    scale_factor = 2.0 / radius
    mesh = mesh.apply_transform(translation_matrix)
    mesh.apply_scale(scale_factor)

    angles_rad = np.radians(config['render']['euler'])
    rotation = R.from_euler('xyz', angles_rad).as_matrix()
    mesh.apply_transform(np.vstack((
        np.hstack((rotation, [[0], [0], [0]])),
        [0, 0, 0, 1]
    )))

    mesh.export(os.path.join(data_dir, "mesh_normalized.glb"))

    face_normal = mesh.face_normals

    ### sample pointcloud for superpoint generation
    if type(mesh.visual) == trimesh.visual.color.ColorVisuals:
        mesh.visual = mesh.visual.to_texture()
        pass
    else:
        if type(mesh.visual.material) == trimesh.visual.material.SimpleMaterial:
            pass
        else:
            mesh.visual.material = mesh.visual.material.to_simple()
    points_sampled, face_indices, points_rgb = trimesh.sample.sample_surface(mesh, 800000, sample_color=True)
    if type(points_rgb) == type(None):
        points_rgb = np.ones_like(points_sampled) * 128
    points_normal = face_normal[face_indices]
    save_colored_pc(os.path.join(data_dir, "sampled_points.ply"), points_sampled, points_rgb)
    np.save(os.path.join(data_dir, "points_normal.npy"), points_normal)
    np.save(os.path.join(data_dir, "points_xyz.npy"), points_sampled)
    np.save(os.path.join(data_dir, "points_color.npy"), points_rgb)
    np.save(os.path.join(data_dir, "points_face_idx.npy"), face_indices)
    
    ### load the mesh into genesis
    gs.init(backend=gs.gpu)
    if not config['render']['use_luisa']:
        scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(),
            vis_options=gs.options.VisOptions(
                show_world_frame=False,
                segmentation_level='entity',
                background_color = (0.5,0.5,0.5),
                ambient_light = (1.,1.,1.),
                shadow=False,
            ),
            show_viewer=False,
        )

        obj = scene.add_entity(
            gs.morphs.Mesh(
                file=mesh_path, 
                fixed=True, 
                collision=False,
                pos=-center*scale_factor,
                scale=(scale_factor, scale_factor, scale_factor),
                euler=config['render']['euler']
            ),
        )
        
    else:
        fa = 2.5
        scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(),
            vis_options=gs.options.VisOptions(
                show_world_frame=False,
                segmentation_level='entity',
                background_color = (1.,1.,1.),
                ambient_light = (1.,1.,1.),
                shadow=False,
            ),
            show_viewer=False,
            renderer=gs.renderers.RayTracer(
                logging_level='info',
                lights=[{'pos': (-4.5*fa, 4.5*fa, 3.0*fa), 'color': (255, 255, 255), 'radius': 0.8, 'intensity': 1.0},
                {'pos': (4.5*fa, 4.5*fa, 3.0*fa), 'color': (255, 255, 255), 'radius': 0.8, 'intensity': 0.6},
                {'pos': (4.5*fa, 0*fa, -3.0*fa), 'color': (255, 255, 255), 'radius': 0.8, 'intensity': 1.0}],
                env_surface=gs.surfaces.Emission(
                    emissive_texture=gs.textures.ColorTexture(
                        color=(0.5,0.5,0.5),
                    ),
                ),
            )
        )

        obj = scene.add_entity(
            gs.morphs.Mesh(
                file=mesh_path, 
                fixed=True, 
                collision=False,
                pos=-center*scale_factor,
                scale=(scale_factor, scale_factor, scale_factor),
                euler=config['render']['euler']
            ),
        )

    ### render
    camera_list = []
    highres_camera_list = []
    for i in range(len(camera_poses_render)):
        extrinsics = np.linalg.inv(camera_poses_render[i])
        print(extrinsics[0,3], extrinsics[1,3], extrinsics[2,3])
        cam = scene.add_camera(
            res=(H, W),
            pos=(extrinsics[0,3], extrinsics[1,3], extrinsics[2,3]),
            lookat=(0, 0, 0),
            up=(0, 1, 0),
            fov=60,
            GUI=False,
        )
        camera_list.append(cam)

        cam = scene.add_camera(
            res=(4096, 4096),
            pos=(extrinsics[0,3], extrinsics[1,3], extrinsics[2,3]),
            lookat=(0, 0, 0),
            up=(0, 1, 0),
            fov=60,
            GUI=False,
        )
        highres_camera_list.append(cam)

    scene.build()
    scene.step()

    os.makedirs(os.path.join(data_dir, "render"), exist_ok=True)

    for i in range(len(camera_poses_render)):
        extrinsics = camera_poses_render[i]
        cam = camera_list[i]
        rgb_arr, depth_arr, seg_arr, normal_arr = cam.render(rgb=True, depth=True, segmentation=False, normal=True)
        cv2.imwrite(os.path.join(data_dir, "render", f"rgb_{i}.png"), np.flip(rgb_arr, axis=-1))
        cv2.imwrite(os.path.join(data_dir, "render", f"normal_{i}.png"), np.flip(rgb_arr, axis=-1))
        np.save(os.path.join(data_dir, "render", f"depth_{i}.npy"), depth_arr)
        background_mask = depth_arr >= 99
        background_mask_img = np.stack([background_mask, background_mask, background_mask], axis=-1).astype(float) *  255
        background_mask_img = 255 - background_mask_img
        cv2.imwrite(os.path.join(data_dir, "render", f"mask_{i}.png"), background_mask_img)
        cv2.imwrite(os.path.join(data_dir, "render", f"normal_{i}.png"), normal_arr)

        cam_highres = highres_camera_list[i]
        rgb_arr, depth_arr, seg_arr, normal_arr = cam_highres.render(rgb=True, depth=True, segmentation=False, normal=True)
        cv2.imwrite(os.path.join(data_dir, "render", f"rgb_highres_{i}.png"), np.flip(rgb_arr, axis=-1))
