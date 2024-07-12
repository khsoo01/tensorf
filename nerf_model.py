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

        self.fc1 = nn.ModuleList()
        self.fc1.append(nn.Sequential(nn.Linear(6*l_pos, hidden1),
                                      nn.ReLU()))
        for _ in range(7):
            self.fc1.append(nn.Sequential(nn.Linear(hidden1, hidden1),  
                                          nn.ReLU()))
        
        self.fc2 = nn.Linear(hidden1, hidden1)
        self.fc3 = nn.Sequential(nn.Linear(hidden1 + 6*l_dir, hidden2),
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

        x = self.pos_enc(input_pos)
        for fc in self.fc1:
            x = fc(x)

        output_density = self.out_density(x)
        
        x = self.fc2(x)
        x = torch.cat((self.dir_enc(input_dir), x), -1)
        x = self.fc3(x)

        output_color = self.out_color(x)
        
        output = torch.cat((output_color, output_density), -1)
        return output


def save_model (path: str, model: NerfModel, num_iter: int):
    model_state = {
        'state_dict': model.state_dict(),
        'l_pos': model.l_pos,
        'l_dir': model.l_dir,
        'hidden1': model.hidden1,
        'hidden2': model.hidden2,
        'num_iter': num_iter
    }
    torch.save(model_state, path)

def load_model (path: str) -> NerfModel:
    if os.path.exists(path):
        model_state = torch.load(path)

        l_pos = model_state['l_pos']
        l_dir = model_state['l_dir']
        hidden1 = model_state['hidden1']
        hidden2 = model_state['hidden2']

        model = NerfModel(l_pos, l_dir, hidden1, hidden2)
        model.load_state_dict(model_state['state_dict'])

        num_iter = model_state['num_iter']
    else:
        model = NerfModel()
        num_iter = 0

    return model, num_iter
