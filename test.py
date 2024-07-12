from nerf_model import NerfModel, load_model
from nerf_types import NerfDataset
from nerf_utils import sample_coarse, render
from load_blender import load_blender

import torch
from torchvision.transforms.functional import to_pil_image

import configparser
import os

def test():
    # Load config
    config = configparser.ConfigParser()
    config.read('config.txt')
    config = dict(config['general']) | dict(config['test'])

    model_path = config['model_path']
    dataset_type = config['dataset_type']
    dataset_path = config['dataset_path']
    num_sample_coarse = int(config['num_sample_coarse'])
    
    save_gt = (config['save_gt'] == 'True')
    output_path = config['output_path']

    # Load dataset
    if dataset_type == 'blender':
        dataset = load_blender(dataset_path)['test']
    else:
        print('Invalid dataset type. Aborting.')
        exit(0)

    # Load dataset arguments
    args = dataset.args
    W = args.width
    H = args.height
    sample_near = args.near
    sample_far = args.far

    # Start testing
    model, _ = load_model(model_path)
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=H*W, shuffle=False)

    for batch_index, batch in enumerate(dataloader):
        inputs, outputs = batch
        sample, t_sample = sample_coarse(inputs, num_sample_coarse, sample_near, sample_far)

        model_outputs = model(sample)
        pixels = render(t_sample, model_outputs)

        image = to_pil_image(pixels.detach().reshape((H, W, 3)).numpy())
        image.save(os.path.join(output_path, f'output{batch_index}.png'), format='PNG')

        if save_gt:
            image = to_pil_image(outputs.detach().reshape((H, W, 3)).numpy())
            image.save(os.path.join(output_path, f'output{batch_index}-gt.png'), format='PNG')

        print(f'Image saved: {batch_index}')

if __name__ == '__main__':
    test()
