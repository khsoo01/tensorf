import torch

from math import tan

def get_rays (W: int, H: int, fov: float, c2w: torch.tensor) -> torch.tensor:
    focal = 0.5 * W / tan(0.5 * fov)

    grid_x, grid_y = torch.meshgrid(torch.arange(0, W, dtype = torch.float32),
                                    torch.arange(0, H, dtype = torch.float32), indexing='xy')

    # in camera homogeneous coordinates
    dirs = torch.stack([(grid_x - W*0.5) / focal, (grid_y - H*0.5) / focal, -torch.ones_like(grid_x), torch.zeros_like(grid_x)], -1).reshape(H*W, 4)
    org = torch.tensor([0, 0, 0, 1], dtype = torch.float32)

    # in world coordinates
    dirs = (dirs @ c2w.T)[:, :3]
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True) # normalize
    org = (org @ c2w.T)[:3]
    orgs = org.repeat(H*W, 1)
    
    rays = torch.cat((orgs, dirs), dim=1)

    return rays

def sample_coarse (rays: torch.tensor, num_sample: int, near: float, far: float) -> torch.tensor:
    orgs = rays[..., :3].unsqueeze(-2) # shape: (..., 1, 3)
    dirs = rays[..., 3:].unsqueeze(-2) # shape: (..., 1, 3)

    intervals = torch.linspace(near, far, num_sample+1)
    midpoints = (intervals[1:] + intervals[:-1]) / 2

    t_samples = midpoints + (torch.rand_like(midpoints) - 0.5) * (far - near) / num_sample
    t_samples = t_samples.unsqueeze(-1) # shape: (num_sample, 1)
    pos_samples = orgs + t_samples * dirs # shape: (..., num_sample, 3)
    dir_samples = torch.broadcast_to(dirs, pos_samples.shape) # shape: (..., num_sample, 3)

    samples = torch.cat([pos_samples, dir_samples], -1) # shape: (..., num_sample, 6)

    return samples, t_samples

def render (t_samples: torch.tensor, color_d: torch.tensor) -> torch.tensor:
    color = color_d[..., :3] # shape: (..., num_sample, 3)
    density = color_d[..., 3:] # shape: (..., num_sample, 1)

    delta = t_samples[1:] - t_samples[:-1]
    broadcasted_inf = torch.broadcast_to(torch.tensor([1e10]), delta[..., :1, :1].shape)
    delta = torch.cat([delta, broadcasted_inf], -2) # append inf = 1e10; shape: (num_sample, 1)

    alpha = 1-torch.exp(-density*delta) # shape: (..., num_sample, 1)
    T = torch.cumprod(1.0 - alpha + 1e-10, dim=-2)
    broadcasted_one = torch.broadcast_to(torch.tensor([1.0]), T[...,:1,:1].shape)
    T = torch.cat([broadcasted_one, T[...,:-1,:1]], -2) # Equivalent with exclusive=True

    out_color = (T * alpha * color).sum(dim=-2) # shape: (..., 3)
    return out_color
