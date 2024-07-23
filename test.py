from nerf_model import NerfModel, load_model
from nerf_types import NerfDataset
from nerf_utils import sample, render
from load_blender import load_blender

import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image

import configparser
import sys
import os

def test(config_path: str = None):
    config_paths = ['config_default.txt']
    if config_path is not None:
        config_paths.append(config_path)

    # Load config
    config = configparser.ConfigParser()
    config.read(config_paths)
    config = dict(config['general']) | dict(config['test'])

    model_path = config['model_path']
    dataset_type = config['dataset_type']
    dataset_path = config['dataset_path']
    resolution_ratio = float(config['resolution_ratio'])
    batch_size = int(config['batch_size'])
    num_sample = int(config['num_sample'])
    
    save_gt = (config['save_gt'] == 'True')
    make_gif = (config['make_gif'] == 'True')
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
        dataset = load_blender(dataset_path, resolution_ratio)['test']
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

    # Start testing
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
    
    model, _ = load_model(model_path)
    model = model.to(device)
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=H*W, shuffle=False)

    # Evaluate image from rays using model
    def eval_image(rays: torch.tensor):
        samples, t_samples = sample(rays, num_sample, sample_near, sample_far)
        # Normalize sample positions to be in [-1, 1] (for positional encoding)
        samples[..., :3] /= max_coord

        model_outputs = model(samples.to(device)).to(cpu)
        image, _ = render(t_samples, model_outputs)

        return image

    images = []
    images_gt = []

    for batch_index, batch in enumerate(dataloader):
        inputs, outputs_gt = batch

        outputs = []
        for start_index in range(0, H*W, batch_size):
            end_index = min(start_index + batch_size, H*W)
            with torch.no_grad():
                outputs.append(eval_image(inputs[start_index:end_index]))
        
        outputs = torch.cat(outputs, 0)
        
        mse_loss = nn.MSELoss()
        mse = mse_loss(outputs, outputs_gt)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = (-10.0 * torch.log(mse) / torch.log(torch.Tensor([10.0]))).item()

        image = to_pil_image(outputs.detach().reshape((H, W, 3)).numpy())
        images.append(image)
        image.save(os.path.join(output_path, f'output{batch_index}.png'), format='PNG')

        if save_gt:
            image = to_pil_image(outputs_gt.detach().reshape((H, W, 3)).numpy())
            images_gt.append(image)
            image.save(os.path.join(output_path, f'output{batch_index}-gt.png'), format='PNG')

        print(f'Image saved: {batch_index}. (psnr={psnr})')

    if make_gif:
        images[0].save(
            os.path.join(output_path, 'animated.gif'),
            save_all=True,
            append_images=images[1:],
            duration=0.1,  # Duration between frames in milliseconds
            loop=0  # Loop forever
        )

        if save_gt:
            images_gt[0].save(
                os.path.join(output_path, 'animated-gt.gif'),
                save_all=True,
                append_images=images_gt[1:],
                duration=0.1,
                loop=0
            )

if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    test(config_path)
