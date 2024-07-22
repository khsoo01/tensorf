from nerf_model import NerfModel, load_model, save_model
from nerf_types import NerfDataset
from nerf_utils import get_lr, sample, render
from load_blender import load_blender

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.functional import to_pil_image

import configparser
import sys
import os

def train(config_path: str = None):
    config_paths = ['config_default.txt']
    if config_path is not None:
        config_paths.append(config_path)

    # Load config
    config = configparser.ConfigParser()
    config.read(config_paths)
    config = dict(config['general']) | dict(config['train'])

    model_path = config['model_path']
    dataset_type = config['dataset_type']
    dataset_path = config['dataset_path']
    resolution_ratio = float(config['resolution_ratio'])
    batch_size = int(config['batch_size'])

    num_iter = int(config['num_iteration'])
    model_save_interval = int(config['model_save_interval'])
    image_save_interval = int(config['image_save_interval'])
    save_image = bool(config['save_image'])
    output_path = config['output_path']

    print(f'Loaded config from {config_paths}.')

    # Make sure that directories in file paths exist
    def create_directories(path: str):
        extension = os.path.splitext(path)[1]
        if len(extension) > 0:
            directory_path = os.path.dirname(path)
        else:
            directory_path = path
        try:
            os.makedirs(directory_path, exist_ok=True)
        except Exception as e:
            print(f"Error occurred while makedirs: {e}")

    for path in [model_path, output_path]:
        create_directories(path)

    # Load dataset
    if dataset_type == 'blender':
        dataset = load_blender(dataset_path, resolution_ratio)['train']
    else:
        print('Invalid dataset type. Aborting.')
        exit(0)

    # Load dataset arguments
    args = dataset.args
    W = args.width
    H = args.height
    sample_near = args.near
    sample_far = args.far
    max_coord = args.max_coord

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

    # TODO Model, DataLoader, Optimizer declaration
    model = None
    dataloader = None
    optimizer = None

    # Setup example input and ground truth image from the input
    if save_image:
        example_input, example_output_gt = dataset[0:H*W]
        example_image_gt = to_pil_image(example_output_gt.detach().reshape((H, W, 3)).numpy())
        example_image_gt.save(os.path.join(output_path, 'train-gt.png'), format='PNG')

    # Evaluate image from rays using model
    def eval_image(rays: torch.tensor):
        # TODO Evaluate pixel value of rays
        return None

    def train_batch(batch: torch.tensor):
        nonlocal cur_iter

        print(f'Training: iteration {cur_iter}.')
        # Evaluate image from batch input and backpropagate loss
        inputs, outputs = batch
        
        # TODO Evaluate model output and loss
        loss = None
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        cur_iter += 1

        # Save model
        if cur_iter % model_save_interval == 0:
            # TODO Save model
            pass

        # Save example image
        if save_image and cur_iter % image_save_interval == 0:
            example_output = []
            for start_index in range(0, H*W, batch_size):
                end_index = min(start_index + batch_size, H*W)
                inputs = example_input[start_index:end_index]
                with torch.no_grad():
                    image = eval_image(inputs)
                    example_output.append(image)

            example_output = torch.cat(example_output, 0)

            example_image = to_pil_image(example_output.detach().reshape((H, W, 3)).numpy())
            example_image.save(os.path.join(output_path, f'train-iter{cur_iter}.png'), format='PNG')
            print('Image saved.')

    while cur_iter < num_iter:
        for batch in dataloader:
            train_batch(batch)
            if cur_iter >= num_iter:
                break

if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    train(config_path)
