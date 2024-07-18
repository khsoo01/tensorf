from nerf_types import NerfDataset, NerfDatasetArgs
from nerf_utils import get_rays

import torch
from torchvision.transforms.functional import pil_to_tensor

from PIL import Image

from typing import Dict
import os
import glob
import json

def parse_blender (fov: float, c2w: torch.tensor, image: torch.tensor) -> NerfDataset:
    W = image.shape[2]
    H = image.shape[1]
    
    rays = get_rays(W, H, fov, c2w)

    image_rgb = image[:3, :, :]
    image_alpha = image[3, :, :].reshape((1, H, W))
    colors = (image_rgb * image_alpha).permute(1, 2, 0)[:, :, :3].reshape(H*W, 3)

    args = NerfDatasetArgs(width=W, height=H, near=2.0, far=6.0, max_coord=4.0)
    return NerfDataset(args=args, rays=rays, colors=colors)

def load_blender (base_path: str, res_ratio: float = 1.0) -> Dict[str, NerfDataset]:
    print(f"Loading blender dataset from {base_path}.")

    search_pattern = os.path.join(base_path, 'transforms_*.json')
    json_files = glob.glob(search_pattern)

    result: Dict[str, NerfDataset] = {}

    for json_file in json_files:
        try:
            dataset = None

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

                    parsed = parse_blender(fov, c2w, image)
                    if dataset is None:
                        dataset = parsed
                    else:
                        dataset.append(parsed)

            # ".../transforms_train.json" -> "train"
            short_name = json_file.replace('_', '.').split(sep = '.')[-2]

            result[short_name] = dataset
        
        except Exception as e:
            print(f'Error occurred in file {json_file}: {e}')

    print("Finished dataset loading.")
    return result

if __name__ == '__main__':
    data = load_blender("./NeRF_Data/nerf_synthetic/chair", 0.1)
    for key in data:
        print(f'{key}:')
        print(f'Shape of rays: {data[key].rays.shape}')
        print(f'Shape of colors: {data[key].colors.shape}')
