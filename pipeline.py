import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import torch
import pytorch3d
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imageio.v3 as imageio

from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PerspectiveCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    #SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

from shading import SoftPhongShader
from pytorch3d.transforms import Scale
from depth_renderer import MeshRenderWithDepth
from utils import image_grid, avg_pool


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file-name", type=str, default = 'test.obj')
parser.add_argument("--output-dir", type=str, default = "output")
parser.add_argument("--use-gpu", type=int, default = 1)
parser.add_argument("--num-views", type=int, default = 1)
parser.add_argument("--no-depth", type=int, default = 0)
parser.add_argument("--aa-factor", type=int, default = 1, help = "anti-alias factor")
parser.add_argument("--radius", type=float, default = 1.2)
parser.add_argument("--scale", type=float, default = 0.9)
args = parser.parse_args()

if args.use_gpu:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    
else:
    device = torch.device("cpu")    

## load obj file
DATA_DIR = './data/cow_mesh'
obj_filename = os.path.join(DATA_DIR, "cow.obj")
mesh = load_objs_as_meshes([obj_filename], device=device)
# prit(mesh.texture)

# verts, faces, aux = load_obj([args.file_name], load_textures = True, device = device)
# verts, faces, aux = load_obj(f_obj, load_textures=True)
# print(aux.texture_images)
# print(aux.material_colors)
mesh = mesh.extend(args.num_views)

## scale obj
verts = mesh.verts_packed()
N = verts.shape[0]
center = verts.mean(0)
scale = max((verts - center).abs().max(0)[0])
mesh.offset_verts_(-(center.expand(N, 3)))
mesh.scale_verts_((1.0 / float(scale) * args.scale))

## create renderer
### Initialize a camera
elev = torch.rand((args.num_views,)) * 90 - 30
azim = torch.rand((args.num_views,)) * 360
# # elev = 

# R, T = look_at_view_transform(args.radius, elev, azim)
# fovy = torch.tensor(np.arctan(32 / 2 / 35) * 2)
# cameras = FoVPerspectiveCameras(device = device, R = R, T = T, 
#     fov = fovy ) #,in_ndc=False)
rotation , translation = look_at_view_transform(5 , 
                                                elev, 
                                                azim,
                                                device = 'cuda')

cameras = FoVPerspectiveCameras(R = rotation , 
                               T = translation ,
                               device = 'cuda')


## Creating the Rasterization settings ##

rasterization_settings = RasterizationSettings(image_size = (256 , 256) , 
                                               blur_radius = 0. , 
                                               faces_per_pixel = 1)


### Rasterization
raster_settings = RasterizationSettings(
    image_size = 256 * args.aa_factor,
    blur_radius = 0.0,
    faces_per_pixel=1
)
rasterizer = MeshRasterizer(
    cameras = cameras,
    raster_settings = raster_settings
)

### Add a point light
lights = PointLights(device = device, location = [0., 0., 0.5])

### Shader
shader = SoftPhongShader(
    device = device,
    cameras = cameras,
    lights = lights
)

if args.no_depth:
    renderer = MeshRenderer(
        rasterizer = rasterizer,
        shader = shader
    )
    images = renderer(mesh)
else:
    renderer = MeshRenderWithDepth(
        rasterizer = rasterizer,
        shader = shader
    )
    images, depths = renderer(mesh)



plt.figure(figsize = (10, 10))
# print(images.size())
# print(images)
print(depths.size())

os.makedirs(args.output_dir, exist_ok = True)
for i in range(args.num_views):
    plt.imsave(f'{args.output_dir}/rgb_{i:04}.png', images[i, ..., :3].cpu().numpy())
    if not args.no_depth:
        cv2.imwrite(f'{args.output_dir}/depth_{i:04}.exr', depths[i, ..., 0].cpu().numpy())
        cv2.imwrite(f'{args.output_dir}/depth_{i:04}.png', depths[i, ..., 0].cpu().numpy())

# image_grid(images.cpu().numpy(), rows=4, cols=5, rgb=True)


