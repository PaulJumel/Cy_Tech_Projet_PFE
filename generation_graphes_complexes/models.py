import torch
import torch.nn as nn

 # Generator architecture depending on the type of graph
class Generator(nn.Module):
    def __init__(self, n_nodes, type_graphs=1):
        super(Generator, self).__init__()
        self.type_graphs = type_graphs
        self.n_nodes = n_nodes

        if self.type_graphs in [1,4,5]:
            self.fc = nn.Sequential(
                nn.Linear(n_nodes * n_nodes, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, n_nodes * n_nodes),
                nn.Sigmoid()    
            )
        elif self.type_graphs == 2:
            self.fc = nn.Sequential(
                nn.Linear(n_nodes * n_nodes, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.7),
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(512, n_nodes * n_nodes)
            )
        elif self.type_graphs == 3:
            self.fc = nn.Sequential(
                nn.Linear(n_nodes * n_nodes, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, n_nodes * n_nodes),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Type de graphe non pris en charge : {type_graphs}")
        
    def forward(self, z):
        # Project latent vector into adjacency matrix space
        adj_matrix = self.fc(z).view(-1, self.n_nodes, self.n_nodes)

        # Ensure symmetry and remove self-loops
        adj_matrix = (adj_matrix + adj_matrix.transpose(1, 2)) / 2
        adj_matrix = adj_matrix * (1 - torch.eye(self.n_nodes)).to(adj_matrix.device)

        # Special handling for cactus graphs with differentiable binarization
        if self.type_graphs == 2:
            adj_matrix = torch.sigmoid(adj_matrix)
            x = (adj_matrix > 0.5).float() + adj_matrix - adj_matrix.detach()
        else:
            x = adj_matrix

        return x
    

# Discriminator architecture varies by graph type
class Discriminator(nn.Module):
    def __init__(self, n_nodes, type_graphs=1):
        super(Discriminator, self).__init__()
        self.type_graphs = type_graphs
        self.n_nodes = n_nodes

        if self.type_graphs in [1,4,5]:
            self.fc = nn.Sequential(
                nn.Linear(n_nodes * n_nodes, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 1),
                nn.Sigmoid() 
            )
        elif self.type_graphs in [2,3]:
            self.fc = nn.Sequential(
                nn.Linear(n_nodes * n_nodes, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Type de graphe non pris en charge : {type_graphs}")

    def forward(self, x):
        # Flatten adjacency matrix before classification
        x = x.view(-1, self.n_nodes * self.n_nodes)
        return self.fc(x)