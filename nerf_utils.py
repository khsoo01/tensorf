import torch

from math import tan

def get_rays (W: int, H: int, fov: float, c2w: torch.tensor) -> torch.tensor:
    focal = 0.5 * W / tan(0.5 * fov)

    grid_x, grid_y = torch.meshgrid(torch.arange(0, W, dtype = torch.float32),
                                    torch.arange(0, H, dtype = torch.float32), indexing='xy')

    # in camera homogeneous coordinates
    dirs = torch.stack([(grid_x - W*0.5) / focal, -(grid_y - H*0.5) / focal, -torch.ones_like(grid_x), torch.zeros_like(grid_x)], -1).reshape(H*W, 4)
    org = torch.tensor([0, 0, 0, 1], dtype = torch.float32)

    # in world coordinates
    dirs = (dirs @ c2w.T)[:, :3]
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True) # normalize
    org = (org @ c2w.T)[:3]
    orgs = org.repeat(H*W, 1)
    
    rays = torch.cat((orgs, dirs), dim=1)

    return rays

def render (t_samples: torch.tensor, color_d: torch.tensor) -> torch.tensor:
    color = color_d[..., :3] # shape: (num_rays, num_sample, 3)
    density = color_d[..., 3:] # shape: (num_rays, num_sample, 1)

    delta = t_samples[..., 1:, :] - t_samples[..., :-1, :] # shape: (num_rays, num_sample-1, 1)
    infs = 1e10 * torch.ones_like(delta[..., :1, :1])
    delta = torch.cat([delta, infs], -2) # append inf = 1e10; shape: (num_rays, num_sample, 1)

    alpha = 1-torch.exp(-density*delta) # shape: (num_rays, num_sample, 1)
    T = torch.cumprod(1.0 - alpha + 1e-10, dim=-2) # shape: (num_rays, num_sample, 1)
    ones = torch.ones_like(T[..., :1, :1])
    T = torch.cat([ones, T[...,:-1,:1]], -2) # Equivalent with exclusive=True

    weight = T * alpha # shape: (num_rays, num_sample, 1)
    out_color = (weight * color).sum(dim=-2) # shape: (num_rays, 3)
    weight = weight.squeeze(-1) # shape: (num_rays, num_sample)

    return out_color, weight
