import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

import os

# Generate paramter with given shape using kaiming init
def make_param(shape):
    param_tensor = torch.empty(shape, dtype=torch.float32)
    nn.init.kaiming_uniform_(param_tensor, nonlinearity='relu')
    return nn.Parameter(param_tensor)

class PositionalEncodingLayer(nn.Module):
    def __init__(self, length):
        super(PositionalEncodingLayer, self).__init__()
        self.length = length
        self.coeffs = 2.0 ** torch.linspace(0, length - 1, length) * torch.pi

    def forward (self, input):
        #  input : [..., X]
        # output : [..., 2*L*X]
        output = []
        for coeff in self.coeffs:
            output.append(torch.sin(coeff * input))
            output.append(torch.cos(coeff * input))
        output = torch.cat(output, -1)

        return output

class TensorfVM(nn.Module):
    def __init__(self, N, R):
        super(TensorfVM, self).__init__()

        self.vectors = make_param((3, R, N, 1))
        self.matrices = make_param((3, R, N, N))

    def reset_grid_size(self, grid_size):
        self.vectors = nn.Parameter(F.interpolate(self.vectors.data, size=[grid_size, 1], mode='bilinear'))
        self.matrices = nn.Parameter(F.interpolate(self.matrices.data, size=[grid_size, grid_size], mode='bilinear'))

    def forward(self, input):
        #  input: [B, S, 3] 3D cartesian coordinate (X, Y, Z)
        # output: [B, S, 3*R] Feature
        B = input.shape[0]
        S = input.shape[1]

        # shape = [3, B, S, 2]
        mat_coord = torch.empty(3, B, S, 2, dtype=input.dtype, device=input.device)
        vec_coord = torch.empty(3, B, S, 2, dtype=input.dtype, device=input.device)

        for i in range(3):
            mat_coord[i, :, :, 0] = input[:, :, (i+1)%3]
            mat_coord[i, :, :, 1] = input[:, :, (i+2)%3]
            vec_coord[i, :, :, 0] = input[:, :, i]
            vec_coord[i, :, :, 1] = 0

        # shape = [3, R, B, S]
        mat_output = F.grid_sample(self.matrices, mat_coord, align_corners=True)
        vec_output = F.grid_sample(self.vectors, vec_coord, align_corners=True)

        # shape = [B, S, 3*R]
        output = (mat_output * vec_output).permute(2, 3, 0, 1).view(B, S, -1)

        return output

class TensorfModel(nn.Module):
    def __init__(self, grid_size=300, comp_c=48, comp_d=16, feature_dim=27, hidden_dim=128, pe_dim=2):
        super(TensorfModel, self).__init__()

        self.vm_c = TensorfVM(grid_size, comp_c)
        self.vm_d = TensorfVM(grid_size, comp_d)

        self.feat = nn.Linear(3*comp_c, feature_dim, bias=False)
        self.mlp = nn.Sequential(nn.Linear((1+2*pe_dim)*(3+feature_dim), hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, 3),
                                 nn.Sigmoid())

        self.pe = PositionalEncodingLayer(pe_dim)

    def reset_grid_size(self, grid_size):
        self.vm_c.reset_grid_size(grid_size)
        self.vm_d.reset_grid_size(grid_size)

    def get_param_groups(self, lr_spatial, lr_mlp):
        param_groups = [
            {'params': self.vm_c.parameters(), 'lr': lr_spatial},
            {'params': self.vm_d.parameters(), 'lr': lr_spatial},
            {'params': self.feat.parameters(), 'lr': lr_mlp},
            {'params': self.mlp.parameters(), 'lr': lr_mlp}
        ]
        return param_groups

    def forward(self, input):
        #  input: [B, S, 6] (Position, Direction)
        # output: [B, S, 4] (R, G, B, Density)
        position = input[..., :3]
        direction = input[..., 3:]

        # Evaluate Color
        feature = self.feat(self.vm_c(position))

        mlp_input = [feature, direction, self.pe(feature), self.pe(direction)]
        mlp_input = torch.cat(mlp_input, -1)
        color = self.mlp(mlp_input)

        # Evaluate Density
        density = F.relu(torch.sum(self.vm_d(position), -1, keepdim = True))

        # Concatenate color and density
        output = torch.cat((color, density), -1)

        return output

def save_model(path: str, model: TensorfModel, num_iter: int):
    model_state = {
        'state_dict': model.state_dict(),
        'num_iter': num_iter
    }
    torch.save(model_state, path)

def load_model(path: str) -> (TensorfModel, int):
    if os.path.exists(path):
        model_state = torch.load(path)
        model = TensorfModel()
        model.load_state_dict(model_state['state_dict'])
        num_iter = model_state['num_iter']
    else:
        model = TensorfModel()
        num_iter = 0

    return model, num_iter
