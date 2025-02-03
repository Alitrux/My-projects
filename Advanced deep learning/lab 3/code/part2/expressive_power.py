import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from models import GNN
from utils import sparse_mx_to_torch_sparse_tensor

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
hidden_dim = 32
output_dim = 4
dropout = 0.0
neighbor_aggr = 'mean'
readout = 'mean'


cycle_graphs = list()
for i in range(10, 20):
    cycle_graphs.append(nx.cycle_graph(i))


adj_matrices = [nx.adjacency_matrix(graph) for graph in cycle_graphs]
block_diag_matrix = sp.block_diag(adj_matrices)
adj = sparse_mx_to_torch_sparse_tensor(block_diag_matrix).to(device)

num_nodes = block_diag_matrix.shape[0]
features = np.ones((num_nodes, 1))
features = torch.FloatTensor(features).to(device)

vector = list()
for i, graph in enumerate(cycle_graphs):
    vector.extend([i] * graph.number_of_nodes())
vector = torch.LongTensor(vector).to(device)



# Initialize GNN with mean aggregation and mean readout
gnn_mean_mean = GNN(input_dim=1, hidden_dim=hidden_dim, output_dim=output_dim, 
                    dropout=dropout, neighbor_aggr='mean', readout='mean').to(device)

# Perform a feedforward pass
gnn_mean_mean.eval()
with torch.no_grad():
    output_mean_mean = gnn_mean_mean(features, adj, vector)
print("Mean Aggregation and Mean Readout:", output_mean_mean)

# Initialize GNN with sum aggregation and mean readout
gnn_sum_mean = GNN(input_dim=1, hidden_dim=hidden_dim, output_dim=output_dim, 
                   dropout=dropout, neighbor_aggr='sum', readout='mean').to(device)

# Perform a feedforward pass
gnn_sum_mean.eval()
with torch.no_grad():
    output_sum_mean = gnn_sum_mean(features, adj, vector)
print("Sum Aggregation and Mean Readout:", output_sum_mean)

# Initialize GNN with mean aggregation and sum readout
gnn_mean_sum = GNN(input_dim=1, hidden_dim=hidden_dim, output_dim=output_dim, 
                   dropout=dropout, neighbor_aggr='mean', readout='sum').to(device)

# Perform a feedforward pass
gnn_mean_sum.eval()
with torch.no_grad():
    output_mean_sum = gnn_mean_sum(features, adj, vector)
print("Mean Aggregation and Sum Readout:", output_mean_sum)

# Initialize GNN with sum aggregation and sum readout
gnn_sum_sum = GNN(input_dim=1, hidden_dim=hidden_dim, output_dim=output_dim, 
                  dropout=dropout, neighbor_aggr='sum', readout='sum').to(device)

# Perform a feedforward pass
gnn_sum_sum.eval()
with torch.no_grad():
    output_sum_sum = gnn_sum_sum(features, adj, vector)
print("Sum Aggregation and Sum Readout:", output_sum_sum)




G1 = nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))
G2 = nx.cycle_graph(6)

# Create adjacency matrices for G1 and G2
adj_G1 = nx.adjacency_matrix(G1)
adj_G2 = nx.adjacency_matrix(G2)

# Create a sparse block diagonal matrix containing the adjacency matrices of G1 and G2
block_diag_matrix = sp.block_diag([adj_G1, adj_G2])
adj = sparse_mx_to_torch_sparse_tensor(block_diag_matrix).to(device)

# Create features for the nodes of all graphs in the batch
num_nodes = block_diag_matrix.shape[0]
features = np.ones((num_nodes, 1))
features = torch.FloatTensor(features).to(device)

# Create a vector that indicates the graph to which each node belongs
vector = [0] * G1.number_of_nodes() + [1] * G2.number_of_nodes()
vector = torch.LongTensor(vector).to(device)


# Initialize GNN with sum aggregation and sum readout
gnn_sum_sum = GNN(input_dim=1, hidden_dim=hidden_dim, output_dim=output_dim, 
                  dropout=dropout, neighbor_aggr='sum', readout='sum').to(device)

# Perform a feedforward pass
gnn_sum_sum.eval()
with torch.no_grad():
    output_sum_sum = gnn_sum_sum(features, adj, vector)
print("Sum Aggregation and Sum Readout for G1 and G2:", output_sum_sum)