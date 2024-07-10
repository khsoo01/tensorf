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
    model_save_interval = int(config['model_save_interval'])
    batch_size = int(config['batch_size'])
    learning_rate = float(config['learning_rate'])
    num_sample_coarse = int(config['num_sample_coarse'])

    # Start training
    model = load_model(model_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for batch_index, batch in enumerate(dataloader):
        print(f'Training: batch {batch_index}.')
        inputs, outputs = batch
        sample = sample_coarse(inputs, num_sample_coarse)

        optimizer.zero_grad()

        model_outputs = model(sample)
        loss = nn.MSELoss(render(model_outputs), outputs)
        loss.backward()

        optimizer.step()

        if batch_index % model_save_interval:
            save_model(model_path, model)

if __name__ == '__main__':
    dataset = load_blender('./NeRF_Data/nerf_synthetic/chair', 1/200)['train']
    train(dataset)