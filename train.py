from tensorf_model import TensorfModel, load_model, save_model
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
    num_sample = int(config['num_sample'])

    num_iter = int(config['num_iteration'])
    lr_decay_iter = int(config['lr_decay_iteration'])
    lr_decay_ratio = float(config['lr_decay_ratio'])
    lr_spatial = float(config['lr_spatial'])
    lr_mlp = float(config['lr_mlp'])
    grid_size_start = int(config['grid_size_start'])
    grid_size_end = int(config['grid_size_end'])
    grid_size_steps = eval(config['grid_size_steps'])
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
    else:
        print('Device: cpu')
        device = cpu

    model, cur_iter = load_model(model_path)
    model = model.to(device)
    if cur_iter == 0:
        model.reset_grid_size(grid_size_start)
    
    lr_factor = lr_decay_ratio**(1/lr_decay_iter)
    lr_spatial_init = get_lr(lr_spatial, lr_spatial*lr_decay_ratio, cur_iter, lr_decay_iter)
    lr_mlp_init = get_lr(lr_mlp, lr_mlp*lr_decay_ratio, cur_iter, lr_decay_iter)
    optimizer = optim.Adam(model.get_param_groups(lr_spatial_init, lr_mlp_init))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Setup example input and ground truth image from the input
    if save_image:
        example_input, example_output_gt = dataset[0:H*W]
        example_image_gt = to_pil_image(example_output_gt.detach().reshape((H, W, 3)).numpy())
        example_image_gt.save(os.path.join(output_path, 'train-gt.png'), format='PNG')

    # Evaluate image from rays using model
    def eval_image(rays: torch.tensor):
        samples, t_samples = sample(rays, num_sample, sample_near, sample_far)
        # Normalize sample positions to be in [-1, 1] (for positional encoding)
        samples[..., :3] /= max_coord

        model_outputs = model(samples.to(device)).to(cpu)
        image, _ = render(t_samples, model_outputs)

        return image

    def train_batch(batch: torch.tensor):
        nonlocal cur_iter
        nonlocal optimizer

        print(f'Training: iteration {cur_iter}.')
        # Evaluate image from batch input and backpropagate loss
        inputs, outputs = batch
        
        image = eval_image(inputs)
        mse_loss = nn.MSELoss()
        loss = mse_loss(image, outputs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        cur_iter += 1

        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_factor

        # Update grid size if needed
        if cur_iter in grid_size_steps:
            # Using get_lr since the calculation is same
            new_grid_size = int(get_lr(grid_size_start, grid_size_end, grid_size_steps.index(cur_iter)+1, len(grid_size_steps)))
            model.reset_grid_size(new_grid_size)

            # Optimizer have to be reset due to parameter changes
            lr_spatial_init = get_lr(lr_spatial, lr_spatial*lr_decay_ratio, cur_iter, lr_decay_iter)
            lr_mlp_init = get_lr(lr_mlp, lr_mlp*lr_decay_ratio, cur_iter, lr_decay_iter)
            optimizer = optim.Adam(model.get_param_groups(lr_spatial_init, lr_mlp_init))

            print(f'Grid size updated to {new_grid_size}.')

        # Save model
        if cur_iter % model_save_interval == 0:
            model.to(cpu)
            save_model(model_path, model, cur_iter)
            model.to(device)
            print('Model saved.')

        # Save example image
        if save_image and (cur_iter <= 10 or (cur_iter <= 100 and cur_iter % 10 == 0) or cur_iter % image_save_interval == 0):
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
