import torch
import torch.nn as nn

class ProjectionModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x):
        return self.fc(x)