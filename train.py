from nerf_model import NerfModel, load_model, save_model
from nerf_types import NerfDataset
from nerf_utils import sample_coarse, render
from load_blender import load_blender

import torch
import torch.nn as nn
import torch.optim as optim

import configparser

def train(dataset: NerfDataset):
    # Load config
    config = configparser.ConfigParser()
    config.read('config.txt')
    config = dict(config['general']) | dict(config['train'])

    model_path = config['model_path']
    model_save_epoch = int(config['model_save_epoch'])
    num_epoch = int(config['num_epoch'])
    batch_size = int(config['batch_size'])
    learning_rate = float(config['learning_rate'])
    num_sample_coarse = int(config['num_sample_coarse'])

    # Load dataset arguments
    args = dataset.args
    sample_near = args.near
    sample_far = args.far

    # Start training
    model = load_model(model_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epoch):
        print(f'Training: Epoch {epoch}.')
        for batch in dataloader:
            inputs, outputs = batch
            sample, t_sample = sample_coarse(inputs, num_sample_coarse, sample_near, sample_far)

            optimizer.zero_grad()

            model_outputs = model(sample)
            pixels = render(t_sample, model_outputs)
            loss = nn.MSELoss()(pixels, outputs)
            loss.backward()

            optimizer.step()

        if epoch % model_save_epoch == 0:
            save_model(model_path, model)

if __name__ == '__main__':
    dataset = load_blender('./NeRF_Data/nerf_synthetic/chair', 1/40)['train']
    train(dataset)