import torch 

from torch import nn, Tensor

class MotionConstraints(nn.Module):
    def __init__(self,
                 data_name:str = 'humanml',
                 lambda_vel:float = 0,
                 lambda_foot:float = 0) -> None:
        super().__init__()
        self.data_name = data_name
        self.lambda_vel = lambda_vel
        self.lambda_foot = lambda_foot
        self.njoints = 263 if self.data_name == 'humanml' else 251
        
        self.vel_loss = nn.MSELoss()
        self.foot_loss = nn.MSELoss()
        
    def forward(self, motions:Tensor, motion_ref:Tensor):        
        if self.lambda_vel != 0:
            pass
        
        if self.lambda_foot != 0:
            pass
        
        

import torch
import torch.nn.functional as F

class SimpleMovingAverage(nn.Module):
    def __init__(self, kernel_size):
        super(SimpleMovingAverage, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        # Apply 1D average pooling with the specified kernel size
        x = x.permute(0,2,1)
        smoothed = F.avg_pool1d(x, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        smoothed = smoothed.permute(0,2,1)
        return smoothed

# Example usage
B, L, input_size = 32, 196, 263  # Example shape (B, L, 263)
input_sequence = torch.randint(0, 10, (1, 10, 7)).float()  # Random input tensor

kernel_size = 3  # Size of the moving average window
smoother = SimpleMovingAverage(kernel_size=kernel_size)
smoothed_sequence = smoother(input_sequence)

print("Original Sequence Shape:", input_sequence)
print("Smoothed Sequence Shape:", smoothed_sequence)

