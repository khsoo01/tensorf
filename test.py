from nerf_model import NerfModel, load_model
from nerf_types import NerfDataset
from nerf_utils import sample_coarse, render
from load_blender import load_blender

import torch
from torchvision.transforms.functional import to_pil_image

import configparser
import os

def test(dataset: NerfDataset):
    # Load config
    config = configparser.ConfigParser()
    config.read('config.txt')
    config = dict(config['general']) | dict(config['test'])

    model_path = config['model_path']
    output_path = config['output_path']
    num_sample_coarse = int(config['num_sample_coarse'])

    # Start testing
    model = load_model(model_path)
    W = dataset.width
    H = dataset.height
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=H*W, shuffle=False)

    for batch_index, batch in enumerate(dataloader):
        inputs, _ = batch
        sample = sample_coarse(inputs, num_sample_coarse)

        model_outputs = model(sample)
        pixels = render(model_outputs)

        image = to_pil_image(pixels.numpy())
        image_path = os.path.join(output_path, f'output{batch_index}.png')
        image.save(image_path, format='PNG')
        print(f'Image saved: {image_path}')

if __name__ == '__main__':
    dataset = load_blender('./NeRF_Data/nerf_synthetic/chair', 1/200)['test']
    test(dataset)