import torch.nn as nn
import torch.functional as F
import torch
import torch.optim as optim

import os

# Generate paramter with given shape using kaiming init
def make_param(shape):
    param_tensor = torch.empty(shape, dtype=torch.float32)
    nn.init.kaiming_uniform_(param_tensor, nonlinearity='relu')
    return nn.Parameter(param_tensor)

class TensorfVM(nn.Module):
    def __init__(self, N, R):
        super(TensorfVMScalar, self).__init__()

        self.N = N # Grid size
        self.R = R # Number of matrices / vectors parallelly

        self.vectors = make_param((3, R, N, 1))
        self.matrices = make_param((3, R, N, N))

    def forward(self, input):
        #  input: [B, 3] 3D cartesian coordinate (X, Y, Z)
        # output: [B, 3*R] Feature
        B = input.shape[0]

        # shape = [3, B, 1, 2]
        mat_coord = torch.empty(3, B, 1, 2, dtype=input.dtype, device=input.device)
        vec_coord = torch.empty(3, B, 1, 2, dtype=input.dtype, device=input.device)

        for i in range(3):
            mat_coord[i, :, 0, 0] = input[:, (i+1)%3]
            mat_coord[i, :, 0, 1] = input[:, (i+2)%3]
            vec_coord[i, :, 0, 0] = input[:, i]
            vec_coord[i, :, 0, 1] = 0

        # shape = [3, R, B, 1]
        mat_output = F.grid_sample(self.matrices, mat_coord)
        vec_output = F.grid_sample(self.vectors, vec_coord)

        # shape = [B, 3*R]
        output = (mat_output * vec_output).permute(2, 0, 1, 3).view(B, -1)

        return output

class TensorfModel(nn.Module):
    def __init__(self, grid_size, depth, feature_dim, hidden_dim):
        super(TensorfModel, self).__init__()

        self.vm_c = TensorfVM(grid_size, depth)
        self.vm_d = TensorfVM(grid_size, depth)

        self.feat = nn.Linear(3*depth, feature_dim)
        self.mlp = nn.Sequential(nn.Linear(feature_dim+3, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, 3),
                                 nn.Sigmoid())

    def forward(self, input):
        #  input: [B, 6] (Position, Direction)
        # output: [B, 4] (R, G, B, Density)
        position = input[..., :3]
        direction = input[..., 3:]

        # Evaluate Color
        feature = self.feat(self.vm_c(position))
        mlp_input = torch.cat((feature, direction), -1)
        color = self.mlp(mlp_input)

        # Evaluate Density
        density = torch.sum(self.vm_d(position), 1, keepdim = True)

        # Concatenate color and density
        output = torch.cat((color, density), -1)

        return output
