import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, n_class):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.fc4 = nn.Linear(hidden_dim_3, n_class)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj, idx):
        
        # message passing followed by Relu activation (layer 1)
        x = torch.mm(adj, x_in)
        x = self.relu(self.fc1(x))

        # message passing followed by Relu activation (layer 2)
        x = torch.mm(adj, x)
        x = self.relu(self.fc2(x))
        
        
        # Readout layer
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx)+1, x.size(1), device=x.device)
        out = out.scatter_add_(0, idx, x)

        # fully connected layer followed by Relu activation (layer 1)
        out = self.relu(self.fc3(out))

        # fully connected layer followed by Relu activation (layer 2)
        out = self.fc4(out)
        return F.log_softmax(out, dim=1)
