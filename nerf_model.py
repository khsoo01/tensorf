import torch.nn as nn
import torch.functional as F
import torch
import torch.optim as optim

import os

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

class NerfModel(nn.Module):
    def __init__(self, l_pos = 10, l_dir = 4, hidden1 = 256, hidden2 = 128):
        super(NerfModel, self).__init__()

        self.l_pos = l_pos
        self.l_dir = l_dir
        self.hidden1 = hidden1
        self.hidden2 = hidden2

        self.pos_enc = PositionalEncodingLayer(l_pos)
        self.dir_enc = PositionalEncodingLayer(l_dir)

        self.fc1 = nn.Sequential(nn.Linear(6*l_pos, hidden1),
                                 nn.ReLU(),
                                 nn.Linear(hidden1, hidden1),  
                                 nn.ReLU(),
                                 nn.Linear(hidden1, hidden1),  
                                 nn.ReLU(),
                                 nn.Linear(hidden1, hidden1),  
                                 nn.ReLU(),
                                 nn.Linear(hidden1, hidden1),  
                                 nn.ReLU())

        self.fc2 = nn.Sequential(nn.Linear(6*l_pos + hidden1, hidden1),
                                 nn.ReLU(),
                                 nn.Linear(hidden1, hidden1),  
                                 nn.ReLU(),
                                 nn.Linear(hidden1, hidden1),  
                                 nn.ReLU())
        
        self.fc3 = nn.Linear(hidden1, hidden1)

        self.fc4 = nn.Sequential(nn.Linear(hidden1 + 6*l_dir, hidden2),
                                 nn.ReLU())
        
        self.out_density = nn.Sequential(nn.Linear(hidden1, 1),
                                         nn.ReLU())
        self.out_color = nn.Sequential(nn.Linear(hidden2, 3),
                                       nn.Sigmoid())

    def forward (self, input):
        #  input : [..., 6] (Position, Direction)
        # output : [..., 4] (R, G, B, Density)
        input_pos = input[..., :3]
        input_dir = input[..., 3:]

        pos_e = self.pos_enc(input_pos)
        dir_e = self.dir_enc(input_dir)
        
        x = pos_e
        x = self.fc1(x)
        x = torch.cat((pos_e, x), -1)
        x = self.fc2(x)

        output_density = self.out_density(x)

        x = self.fc3(x)
        x = torch.cat((dir_e, x), -1)
        x = self.fc4(x)

        output_color = self.out_color(x)
        
        output = torch.cat((output_color, output_density), -1)
        return output


def save_model (path: str, model_coarse: NerfModel, model_fine: NerfModel, num_iter: int):
    model_state = {
        'state_dict_coarse': model_coarse.state_dict(),
        'state_dict_fine': model_fine.state_dict(),
        'l_pos': model_fine.l_pos,
        'l_dir': model_fine.l_dir,
        'hidden1': model_fine.hidden1,
        'hidden2': model_fine.hidden2,
        'num_iter': num_iter
    }
    torch.save(model_state, path)

def load_model (path: str) -> (NerfModel, NerfModel, int):
    if os.path.exists(path):
        model_state = torch.load(path)

        l_pos = model_state['l_pos']
        l_dir = model_state['l_dir']
        hidden1 = model_state['hidden1']
        hidden2 = model_state['hidden2']

        model_coarse = NerfModel(l_pos, l_dir, hidden1, hidden2)
        model_coarse.load_state_dict(model_state['state_dict_coarse'])

        model_fine = NerfModel(l_pos, l_dir, hidden1, hidden2)
        model_fine.load_state_dict(model_state['state_dict_fine'])

        num_iter = model_state['num_iter']
    else:
        model_coarse = NerfModel()
        model_fine = NerfModel()
        num_iter = 0

    return model_coarse, model_fine, num_iter
