import torch 

from torch import nn, Tensor
from model.utils.joints import MotionConverter

class MotionConstraints(nn.Module):
    def __init__(self,
                 cfg) -> None:
        super().__init__()
        self.data_name = cfg.DATALOADER.NAME
        if self.data_name not in ['humanml', 'kit']:
            raise TypeError
        
        self.lambda_vel = cfg.lambda_vel
        self.lambda_foot = cfg.lambda_foot
        self.lambda_joint = cfg.lambda_joint
        self.lambda_bone_len = cfg.lambda_bone_len
        self.njoints = 22 if self.data_name == 'humanml' else 21
        
        self.vel_loss = nn.SmoothL1Loss()
        self.foot_loss = nn.SmoothL1Loss()
        self.joint_loss = nn.SmoothL1Loss()
        self.bone_loss = nn.SmoothL1Loss()
        
        self.motion_converter = MotionConverter(cfg=cfg)
        
    def forward(self, motions:Tensor, motions_ref:Tensor):        
        if self.lambda_vel != 0:
            pass
        if self.lambda_foot != 0:
            pass
        if self.lambda_joint != 0:
            pass
        if self.lambda_bone_len != 0:
            pass
    
    def get_bone_length(self, motions: Tensor, motions_ref: Tensor):
        if self.data_name == 'humanml':
            B, L, _ = motions.shape
            local_joint_idx = range(4, 4+(self.njoints-1)*3)
            local_joints = motions[:,:,local_joint_idx].reshape((B,L,self.njoints-1,3))
            local_joints_ref = motions_ref[:,:,local_joint_idx].reshape(
                (B,L,self.njoints-1,3))
            
            
        elif self.data_name == 'kit':
            pass
        else:
            raise TypeError


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

