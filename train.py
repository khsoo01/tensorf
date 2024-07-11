from nerf_model import NerfModel, load_model, save_model
from nerf_types import NerfDataset
from nerf_utils import sample_coarse, render
from load_blender import load_blender

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.functional import to_pil_image

import configparser
import time
import os

def train(dataset: NerfDataset):
    # Load config
    config = configparser.ConfigParser()
    config.read('config.txt')
    config = dict(config['general']) | dict(config['train'])

    model_path = config['model_path']
    output_path = config['output_path']
    model_save_epoch = int(config['model_save_epoch'])
    num_epoch = int(config['num_epoch'])
    batch_size = int(config['batch_size'])
    learning_rate = float(config['learning_rate'])
    num_sample_coarse = int(config['num_sample_coarse'])
    save_image = bool(config['save_image'])

    # Load dataset arguments
    args = dataset.args
    W = args.width
    H = args.height
    sample_near = args.near
    sample_far = args.far

    # Start training
    cpu = torch.device('cpu')

    if torch.cuda.is_available():
        print('Device: cuda')
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        print('Device: mps')
        device = torch.device('mps')
    else:
        print('Device: cpu')
        device = cpu
    
    model = load_model(model_path).to(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if save_image:
        rays, colors = dataset[0:H*W]
        image_gt = to_pil_image(colors.detach().reshape((H, W, 3)).numpy())
        image_path = os.path.join(output_path, 'train-gt.png')
        image_gt.save(image_path, format='PNG')

    for epoch in range(num_epoch):
        print(f'Training: Epoch {epoch}.')

        start_time = time.time()

        for (batch_idx, batch) in enumerate(dataloader):
            print(f'Training: batch {batch_idx}.')
            inputs, outputs = batch
            sample, t_sample = sample_coarse(inputs, num_sample_coarse, sample_near, sample_far)

            optimizer.zero_grad()

            sample = sample.to(device)
            model_outputs = model(sample).to(cpu)
            pixels = render(t_sample, model_outputs)
            loss = nn.MSELoss()(pixels, outputs)
            loss.backward()

            optimizer.step()
        
        end_time = time.time()
        print(f'Time ellapsed: {end_time - start_time} seconds.')

        if epoch % model_save_epoch == 0:
            save_model(model_path, model)

            if save_image:
                sample, t_sample = sample_coarse(rays, num_sample_coarse, sample_near, sample_far)
                model_outputs = model(sample.to(device)).to(cpu)
                pixels = render(t_sample, model_outputs)

                image = to_pil_image(pixels.detach().reshape((H, W, 3)).numpy())
                image_path = os.path.join(output_path, f'train-epoch{epoch}.png')
                image.save(image_path, format='PNG')

                print(f'Image saved.')

if __name__ == '__main__':
    dataset = load_blender('./NeRF_Data/nerf_synthetic/chair', 1/8)['train']
    train(dataset)