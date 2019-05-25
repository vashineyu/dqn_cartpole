#model_torch.py

# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        
        h,w, *_ = input_size
        conv_w_out = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, kernel_size=3, stride=2), 
                                                     kernel_size=3, stride=2), 
                                     kernel_size=3, stride=2)
        conv_h_out = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, kernel_size=3, stride=2), 
                                                     kernel_size=3, stride=2), 
                                     kernel_size=3, stride=2)
        
        linear_input_size = conv_w_out * conv_h_out * 64
        self.head = nn.Linear(linear_input_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

def conv2d_size_out(size, kernel_size = 5, stride = 2):
    return (size - (kernel_size - 1) - 1) // stride  + 1

class DQN_Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN_Linear, self).__init__()
        self.l1 = nn.Linear(input_size, 32)
        self.l2 = nn.Linear(32, 64)
        self.head = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.head(x)