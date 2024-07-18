import torch
from torch.utils.data import Dataset

class NerfDatasetArgs:
    def __init__(self, width: int, height: int, near: float, far: float, max_coord: float):
        self.width = width
        self.height = height
        self.near = near
        self.far = far
        self.max_coord = max_coord

    def __eq__(self, other):
        return (self.width == other.width and
                self.height == other.height and
                self.near == other.near and
                self.far == other.far and
                self.max_coord == other.max_coord)

class NerfDataset(Dataset):
    def __init__(self, args: NerfDatasetArgs, rays = torch.tensor([]), colors = torch.tensor([])):
        self.args = args
        self.rays = rays
        self.colors = colors

    def __len__(self):
        return len(self.rays)
    
    def __getitem__(self, index):
        return self.rays[index], self.colors[index]
    
    def append (self, other): # other: NerfDataset
        assert self.args == other.args
        if len(self) <= 0:
            self.rays = other.rays.clone().detach()
            self.colors = other.colors.clone().detach()
        else:
            self.rays = torch.cat((self.rays, other.rays))
            self.colors = torch.cat((self.colors, other.colors))

    def subset (self, indices):
        selected_rays, selected_colors = self[indices]
        return NerfDataset(args=self.args, rays=selected_rays, colors=selected_colors)
