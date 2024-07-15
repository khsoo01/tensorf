from nerf_model import NerfModel, load_model, save_model
from nerf_types import NerfDataset
from nerf_utils import get_lr, sample, render
from load_blender import load_blender

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.functional import to_pil_image

import configparser
import os

def train():
    # Load config
    config = configparser.ConfigParser()
    config.read('config.txt')
    config = dict(config['general']) | dict(config['train'])

    model_path = config['model_path']
    dataset_type = config['dataset_type']
    dataset_path = config['dataset_path']
    batch_size = int(config['batch_size'])
    num_sample_coarse = int(config['num_sample_coarse'])
    num_sample_fine = int(config['num_sample_fine'])

    num_iter = int(config['num_iteration'])
    lr_start = float(config['learning_rate_start'])
    lr_end = float(config['learning_rate_end'])
    model_save_interval = int(config['model_save_interval'])
    image_save_interval = int(config['image_save_interval'])
    save_image = bool(config['save_image'])
    output_path = config['output_path']

    # Load dataset
    if dataset_type == 'blender':
        dataset = load_blender(dataset_path)['train']
    else:
        print('Invalid dataset type. Aborting.')
        exit(0)

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
    
    model, cur_iter = load_model(model_path)
    model = model.to(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=get_lr(lr_start, lr_end, cur_iter, num_iter))

    # Setup example input and ground truth image from the input
    if save_image:
        example_input, example_output_gt = dataset[0:H*W]
        example_image_gt = to_pil_image(example_output_gt.detach().reshape((H, W, 3)).numpy())
        example_image_gt.save(os.path.join(output_path, 'train-gt.png'), format='PNG')

    # Evaluate image from rays using model
    def eval_image(rays: torch.tensor):
        sample_c, t_sample_c = sample(rays, num_sample_coarse, sample_near, sample_far)

        model_outputs_c = model(sample_c.to(device)).to(cpu)
        coarse, weight = render(t_sample_c, model_outputs_c)

        sample_f, t_sample_f = sample(rays, num_sample_fine, sample_near, sample_far, weight)

        # Concatenate [sample_c, sample_f] and sort by t
        t_sample_f = torch.cat([t_sample_c, t_sample_f], dim=-2)
        sample_f = torch.cat([sample_c, sample_f], dim=-2)
        t_sample_f, indices = torch.sort(t_sample_f, dim=-2)
        indices = torch.broadcast_to(indices, sample_f.shape)
        sample_f = torch.gather(sample_f, -2, indices)

        model_outputs_f = model(sample_f.to(device)).to(cpu)
        fine, _ = render(t_sample_f, model_outputs_f)

        return coarse, fine

    while True:
        for batch in dataloader:
            print(f'Training: iteration {cur_iter}.')
            # Evaluate image from batch input and backpropagate loss
            inputs, outputs = batch
            
            coarse, fine = eval_image(inputs)
            mse_loss = nn.MSELoss()
            loss = mse_loss(coarse, outputs) + mse_loss(fine, outputs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            cur_iter += 1

            # Update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = get_lr(lr_start, lr_end, cur_iter, num_iter)

            # Save model
            if cur_iter % model_save_interval == 0:
                model.to(cpu)
                save_model(model_path, model, cur_iter)
                model.to(device)
                print('Model saved.')

            # Save example image
            if save_image and cur_iter % image_save_interval == 0:
                example_output = []
                for start_index in range(0, H*W, batch_size):
                    end_index = min(start_index + batch_size, H*W)
                    inputs = example_input[start_index:end_index]
                    with torch.no_grad():
                        _, image = eval_image(inputs)
                        example_output.append(image)

                example_output = torch.cat(example_output, 0)

                example_image = to_pil_image(example_output.detach().reshape((H, W, 3)).numpy())
                example_image.save(os.path.join(output_path, f'train-iter{cur_iter}.png'), format='PNG')
                print('Image saved.')

            # Break infinite loop if the iteration is finished
            if cur_iter >= num_iter:
                break
        if cur_iter >= num_iter:
            break

if __name__ == '__main__':
    train()
