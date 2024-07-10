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
    org = (org @ c2w.T)[:3]
    orgs = org.repeat(H*W, 1)
    
    rays = torch.cat((orgs, dirs), dim=1)

    return rays

def sample_coarse (rays: torch.tensor, num_sample: int) -> torch.tensor:
    pass

def render (rays: torch.tensor, points: torch.tensor) -> torch.tensor:
    pass
