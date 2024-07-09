from nerf_types import NerfDataset

import torch
from torchvision.transforms.functional import pil_to_tensor

from PIL import Image

from typing import Dict
from math import tan
import os
import glob
import json

def parse_blender (fov: float, c2w: torch.tensor, image: torch.tensor) -> NerfDataset:
    W = image.shape[2]
    H = image.shape[1]
    focal = 0.5 * W / tan(0.5 * fov)

    grid_x, grid_y = torch.meshgrid(torch.arange(0, W, dtype = torch.float32),
                                    torch.arange(0, H, dtype = torch.float32), indexing='xy')

    # in camera homogeneous coordinates
    dirs = torch.stack([(grid_x - W*0.5) / focal, (grid_y - H*0.5) / focal, -torch.ones_like(grid_x), torch.zeros_like(grid_x)], -1).reshape(H*W, 4)
    org = torch.tensor([0, 0, 0, 1], dtype = torch.float32)

    # in world coordinates
    dirs = (dirs @ c2w.T)[:, :3]
    org = (org @ c2w.T)[:3]
    orgs = org.repeat(H*W, 1)
    
    rays = torch.cat((orgs, dirs), dim=1)

    image_rgb = image[:3, :, :]
    image_alpha = image[3, :, :].reshape((1, H, W))
    colors = (image_rgb * image_alpha).permute(1, 2, 0)[:, :, :3].reshape(H*W, 3)

    return NerfDataset(rays=rays, colors=colors)

def load_blender (base_path: str, res_ratio: float = 1.0) -> Dict[str, NerfDataset]:
    search_pattern = os.path.join(base_path, 'transforms_*.json')
    json_files = glob.glob(search_pattern)

    result: Dict[str, NerfDataset] = {}

    for json_file in json_files:
        try:
            dataset = NerfDataset()

            with open(json_file, 'r') as file:
                json_data = json.load(file)

                fov = json_data['camera_angle_x']
                frames = json_data['frames']

                for frame in frames:
                    image_path = os.path.join(base_path, frame['file_path'] + '.png')
                    pil_image = Image.open(image_path)
                    if res_ratio < 1.0:
                        new_size = (int(pil_image.width * res_ratio), int(pil_image.height * res_ratio))
                        pil_image = pil_image.resize(new_size)
                    image = pil_to_tensor(pil_image).to(torch.float32) / 255.0

                    c2w = torch.tensor(frame['transform_matrix'])

                    dataset.append(parse_blender(fov, c2w, image))

            # ".../transforms_train.json" -> "train"
            short_name = json_file.replace('_', '.').split(sep = '.')[-2]

            result[short_name] = dataset
        
        except Exception as e:
            print(f'Error occurred in file {json_file}: {e}')

    return result

if __name__ == '__main__':
    data = load_blender("./NeRF_Data/nerf_synthetic/chair", 0.1)
    for key in data:
        print(f'{key}:')
        print(f'Shape of rays: {data[key].rays.shape}')
        print(f'Shape of colors: {data[key].colors.shape}')
