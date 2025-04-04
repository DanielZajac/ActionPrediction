import torch
import torch.nn as nn

# class ProjectionModule(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(ProjectionModule, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, output_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(output_dim, output_dim)
#         )
        
#     def forward(self, x):
#         return self.fc(x)

class ProjectionModule(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, dropout=0.3):
        super(ProjectionModule, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.fc(x)