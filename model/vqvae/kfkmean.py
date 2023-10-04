# import torch 
# import random

# from torch import nn, Tensor

# from torch.utils.tensorboard import SummaryWriter

# class KmeanQuantizer(nn.Module):
#     def __init__(self, 
#                  K = 512,
#                  d_input = 263,
#                  device = 'cuda'
#                  ) -> None:
#         super().__init__()
#         self.centroids = nn.Parameter(torch.randn(K, d_input)).to(device)
#         self.num_features = d_input
#         self.num_clusters = K
#         self.device = device
    
#     def forward(self, x):
#         B, L, E = x.shape
#         # Reshape input data to be compatible with centroids
#         x = x.reshape((-1, self.num_features))

#         # Calculate distances between data points and centroids
#         distances = torch.sqrt(torch.sum((x[:, None] - self.centroids) ** 2, dim=2))

#         # Assign each data point to the nearest centroid
#         assignments = torch.argmin(distances, dim=1)
#         # Update centroids based on the assigned data points
#         new_centroids = torch.stack([x[assignments == k].mean(0) for k in range(self.num_clusters)])
#         # Update centroids in-place
#         self.centroids.data = new_centroids.data
#         assigned_centroids = self.centroids.data[assignments,:]
#         assigned_centroids = assigned_centroids.view(B, L, self.num_features)


#         return assignments.reshape((B,L)), assigned_centroids
    
# from torch_kmeans import KMeans
# from torch.utils.data import DataLoader
# from dataloaders.get_data import get_dataset_loader
# import numpy as np

# if __name__ == '__main__':
#     B = 32
#     L = 20
#     E = 263
#     dataloader = get_dataset_loader(
#         name='humanml', batch_size=B, num_frames=None,
#         split='train', hml_mode='train'
#     )
    
#     validloader = get_dataset_loader(
#         name='humanml', batch_size=B, num_frames=None,
#         split='val', hml_mode='train'
#     )
    
#     device = 'cuda'
#     model = KMeans(n_clusters=400,verbose=True).to(device)
    
#     writer = SummaryWriter('log/KFKMEAN.log')
    
#     Loss_fn = nn.MSELoss()
#     for e in range(500):
#         batch_losses = []
#         for batch in dataloader:
#             motion = batch[0].squeeze(2).transpose(-1,-2).to(device)
#             keyframes = motion[:,::10,:].reshape((1, -1, motion.shape[-1]))
#             cluster = model.forward(keyframes)
#             centers = cluster.centers.squeeze()
#             indices = cluster.labels.reshape((-1))
                        
#             cor_centers = centers[indices].reshape((B,L,E))
#             loss = Loss_fn(keyframes.reshape((B,L,E)), cor_centers).cpu().detach().numpy()
#             batch_losses.append(loss)
        
#         for batch in validloader:
#             motion = batch[0].squeeze(2).transpose(-1,-2).to(device)
#             keyframes = motion[:,::10,:].reshape((-1, motion.shape[-1]))
#             distance = torch.sqrt(torch.sum((keyframes[:, None] - centers) ** 2, dim=2))
#             indices = torch.argmin(distance, dim=1)
#             print(indices.shape)
#             assigned = 1
#         print(np.array(batch_losses).mean())
#         writer.add_scalar('MSE loss with Center', scalar_value=loss)
#         torch.save(model.state_dict(),'checkpoints/KFKMEAN.pth')


import time

import numpy as np
from matplotlib import pyplot as plt

from pykeops.numpy import LazyTensor
import pykeops.config



dtype = "float32"

def KMeans(x, K=10, Niter=10, verbose=True):
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means loop:
    # - x  is the point cloud,
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    c = np.copy(x[:K, :])  # Simplistic random initialization
    x_i = LazyTensor(x[:, None, :])  # (Npoints, 1, D)

    for i in range(Niter):
        c_j = LazyTensor(c[None, :, :])  # (1, Nclusters, D)
        D_ij = ((x_i - c_j) ** 2).sum(
            -1
        )  # (Npoints, Nclusters) symbolic matrix of squared distances
        cl = D_ij.argmin(axis=1).astype(int).reshape(N)  # Points -> Nearest cluster

        Ncl = np.bincount(cl).astype(dtype)  # Class weights
        for d in range(D):  # Compute the cluster centroids with np.bincount:
            c[:, d] = np.bincount(cl, weights=x[:, d]) / Ncl
            
    return cl, c

from dataloaders.get_data import get_dataset_loader

if __name__ == '__main__':
    B = 20000
    L = 20
    E = 263
    K = 1024
    dataloader = get_dataset_loader(
        name='humanml', batch_size=B, num_frames=None,
        split='train', hml_mode='train'
    )
    
    validloader = get_dataset_loader(
        name='humanml', batch_size=B, num_frames=None,
        split='val', hml_mode='train'
    )
    
    
    if pykeops.config.gpu_available:
        for batch in dataloader:
            keyframes = batch[0].squeeze(2).transpose(-1,-2).numpy()[:,::10,:]
            keyframes = keyframes.reshape((-1, E))
            cl, c = KMeans(keyframes, K)
            print(cl.shape, c.shape)


# if pykeops.config.gpu_available:
#     N, D, K = 1000000, 100, 1000
#     x = np.random.randn(N, D).astype(dtype)
#     cl, c = KMeans(x, K)

