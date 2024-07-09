import torch
import torch.utils.data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from nerf_types import NerfDataset
from load_blender import load_blender

def visualize_rays(dataset: NerfDataset, arrow_length: float = 1.0):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for batch in dataloader:
        inputs, outputs = batch
        origins = inputs[:, 0:3]
        directions = inputs[:, 3:6]
        colors = outputs

        for i in range(origins.shape[0]):
            origin = origins[i].numpy()
            direction = directions[i].numpy()
            color = colors[i].numpy()

            ax.quiver(
                origin[0], origin[1], origin[2],
                direction[0], direction[1], direction[2],
                length=arrow_length, normalize=True, color=color
            )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

if __name__ == '__main__':
    dataset = load_blender("./NeRF_Data/nerf_synthetic/chair", 1/200)
    # dataset['train'].append(dataset['test'])
    # dataset['train'].append(dataset['val'])
    visualize_rays(dataset['train'])