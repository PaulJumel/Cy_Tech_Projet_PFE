import os
import shutil

import matplotlib.pyplot as plt
import networkx as nx
#import nx_cugraph as nxcg
#import cudf
#import cugraph
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import random

import cliquematch
import math


class Generator(nn.Module):
    def __init__(self, n_nodes):
        super(Generator, self).__init__()
        self.n_nodes = n_nodes
        self.fc = nn.Sequential(
            nn.Linear(n_nodes, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, n_nodes * n_nodes)
        )

    def forward(self, z):
        output = self.fc(z)
        adj_matrix = output.view(-1, self.n_nodes, self.n_nodes)
        adj_matrix = (adj_matrix + adj_matrix.transpose(1, 2)) / 2
        adj_matrix = torch.sigmoid(adj_matrix)
        adj_matrix = adj_matrix * (1 - torch.eye(self.n_nodes)).to(adj_matrix.device)
        #adj_matrix = (adj_matrix > 0.5).float()
        return adj_matrix

class Discriminator(nn.Module):
    def __init__(self, n_nodes):
        super(Discriminator, self).__init__()
        self.n_nodes = n_nodes
        self.fc = nn.Sequential(
            nn.Linear(n_nodes * n_nodes, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, adj_matrix):
        x = adj_matrix.view(-1, self.n_nodes * self.n_nodes)
        return self.fc(x)
    
def maximal_cliquematch_time_loss(adj_matrix):
    batch_size, n_nodes, _ = adj_matrix.size()
    total_loss = torch.zeros(batch_size).to(adj_matrix.device)
    time_loss = torch.zeros(batch_size).to(adj_matrix.device)
    for i in range(batch_size):

        matrice = (adj_matrix[i] > 0.5).float() + adj_matrix[i] - adj_matrix[i].detach()

        G = cliquematch.from_matrix(matrice.detach().cpu().numpy())
        start_time = time.time()
        G.get_max_clique(use_heuristic = False)
        end_time = time.time()
        time_loss[i] = end_time - start_time
    time_loss = -torch.sum(time_loss)
    return time_loss

def maximal_clique_time_loss(adj_matrix):
    batch_size, n_nodes, _ = adj_matrix.size()
    time_loss = torch.zeros(batch_size).to(adj_matrix.device)
    
    
    for i in range(batch_size):
        matrice = (adj_matrix[i] > 0.5).float() + adj_matrix[i] - adj_matrix[i].detach()
        G = nx.from_numpy_array(matrice.detach().cpu().numpy())
        
        start_time = time.time()
        maximal_cliques = list(nx.find_cliques(G))
        end_time = time.time()
        time_loss[i] = end_time - start_time
    
    
    scale_factor = 100.0
    time_loss = torch.log(time_loss + 1e-10)
    final_loss = -scale_factor * torch.sum(time_loss)
    
    return final_loss


def find_mirrors(G, v):
    N_v = set(G.neighbors(v))
    N2_v = set(nx.single_source_shortest_path_length(G, v, cutoff=2).keys()) - N_v - {v}
    mirrors = set()
    for u in N2_v:
        N_v_minus_N_u = N_v - set(G.neighbors(u))
        if all(G.has_edge(x, y) for x in N_v_minus_N_u for y in N_v_minus_N_u if x != y):
            mirrors.add(u)
    return mirrors

def fold_node(G, v):
    neighbors = list(G.neighbors(v))
    anti_edges = [(u, w) for i, u in enumerate(neighbors) for w in neighbors[i+1:] if not G.has_edge(u, w)]
    
    new_nodes = {}
    for u, w in anti_edges:
        new_node = f"{u}_{w}"
        G.add_node(new_node)
        new_nodes[(u, w)] = new_node
    
    for (u, w), new_node in new_nodes.items():
        neighbors_u_w = set(G.neighbors(u)).union(set(G.neighbors(w)))
        for neighbor in neighbors_u_w:
            G.add_edge(new_node, neighbor)
    
    new_node_list = list(new_nodes.values())
    for i in range(len(new_node_list)):
        for j in range(i + 1, len(new_node_list)):
            G.add_edge(new_node_list[i], new_node_list[j])
    
    
    G.remove_nodes_from(neighbors)
    G.remove_node(v)
    
    return G


def mis(G):
    if G.number_of_nodes() == 0:
        return 0, set()
    
    if nx.number_connected_components(G) > 1:
        total_mis = 0
        total_nodes = set()
        components = list(nx.connected_components(G))
        for c in components:
            subgraph = G.subgraph(c).copy()
            subgraph_mis, subgraph_nodes = mis(subgraph)
            total_mis += subgraph_mis
            total_nodes.update(subgraph_nodes)
        return total_mis, total_nodes
    
    
    for v in G.nodes():
        for w in G.nodes():
            if v != w and set(G.neighbors(v)).union({v}).issubset(set(G.neighbors(w)).union({w})):
                H = G.copy()
                print(v,w)
                H.remove_node(w)
                return mis(H)
    
    
    for v in G.nodes():
        neighbors = list(G.neighbors(v))
        if len(neighbors) <= 4:
            anti_edges = 0
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if not G.has_edge(neighbors[i], neighbors[j]):
                        anti_edges += 1
            if anti_edges <= 3:
                H = fold_node(G.copy(), v)
                sub_mis, sub_nodes = mis(H)
                return 1 + sub_mis, sub_nodes.union({v})
    
    
    v = max(G.nodes(), key=G.degree)
    mirrors = find_mirrors(G, v)
    H1 = G.copy()
    H1.remove_node(v)
    H1.remove_nodes_from(mirrors)
    H2 = G.copy()
    H2.remove_node(v)
    H2.remove_nodes_from(list(G.neighbors(v)))
    
    mis1, nodes1 = mis(H1)
    mis2, nodes2 = mis(H2)
    
    if mis1 > 1 + mis2:
        return mis1, nodes1
    else:
        return 1 + mis2, nodes2.union({v})
def generate_erdos_renyi(n, p=0.5):
    
    adj_matrix = np.zeros((n, n))
    
    
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1
                
    return adj_matrix

    

def train(n_nodes=10, n_graphs=256, num_epochs=10_000, batch_size=32, fct=generate_erdos_renyi, loss_algo=maximal_clique_time_loss):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    batch_times = [] 
    g_losses = []
    d_losses = []
    mean_times = []
    var_times = []
    graph_dataset = [fct(n_nodes) for _ in range(n_graphs)]
    
    generator = Generator(n_nodes).to(device)
    discriminator = Discriminator(n_nodes).to(device)
    
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0001, weight_decay=1e-5)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, weight_decay=1e-5)

    for epoch in range(num_epochs):
        epoch_mean_times = []
        epoch_var_times = []
        for i in range(0, n_graphs, batch_size):
            batch_graphs_np = np.array(graph_dataset[i:i+batch_size])
            real_graphs = torch.tensor(batch_graphs_np, dtype=torch.float32).to(device)
            
            z = torch.randn(batch_size, n_nodes).to(device)
            fake_adj = generator(z)
            
           
            times = []
            graphs = []
            for j in range(batch_size):
                single_graph = fake_adj[j].unsqueeze(0)
                time_value = loss_algo(single_graph)
                times.append(float(time_value.item()))
                graphs.append(fake_adj[j].cpu().detach().numpy())
            
            
            sorted_pairs = sorted(zip(times, graphs), key=lambda x: x[0], reverse=True)
            hardest_times, hardest_graphs = zip(*sorted_pairs[:batch_size//2])
            
            
            replace_indices = sorted(range(i, min(i+batch_size, n_graphs)), key=lambda idx: times[idx-i])[:batch_size//2]
            for idx, graph in zip(replace_indices, hardest_graphs):
                graph_dataset[idx] = graph
            
            d_optimizer.zero_grad()
            d_real = discriminator(real_graphs)
            d_fake = discriminator(fake_adj.detach())
            d_loss = -torch.mean(torch.log(d_real) + torch.log(1 - d_fake))
            d_loss.backward()
            d_optimizer.step()
            
            g_optimizer.zero_grad()
            g_fake = discriminator(fake_adj)
            time_algo = loss_algo(fake_adj)
            batch_times.append(float(time_algo.item()))
            
            g_loss = -torch.mean(torch.log(1-g_fake)) + time_algo
            g_loss.backward()
            g_optimizer.step()
            
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            
            epoch_mean_times.append(np.mean(hardest_times))
            epoch_var_times.append(np.var(hardest_times))
        
        
        mean_times.append(np.mean(epoch_mean_times))
        var_times.append(np.mean(epoch_var_times))
        
        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}, Current batch time: {batch_times[-1]}, Hardest time: {hardest_times[0]}')
            torch.save(generator.state_dict(), 'generator_hard.pth')

    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator Losses Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
"""

    plt.figure(figsize=(10, 6))
    plt.plot(batch_times)
    plt.xlabel('Batch Step')
    plt.ylabel('Algorithm Time (seconds)')
    plt.title('Algorithm Time Evolution per Batch')
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), mean_times, label='temps moyen')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Time (seconds)')
    plt.title('Moyenne des graphes ajoutés en fonction du nombre d epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), var_times, label='Variance')
    plt.xlabel('Epoch')
    plt.ylabel('Variance of Time (seconds^2)')
    plt.title('Variance des graphes ajoutés en fonction du nombre d epochs')
    plt.legend()
    plt.grid(True)
    plt.show()"""


n_nodes = 10
train(n_nodes=n_nodes, n_graphs=1024, num_epochs=300
      , batch_size=32, fct=generate_erdos_renyi)

def visualize_generated_graphs(generator, n_samples=5, n_nodes=10):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, n_nodes)
        fake_adj = (generator(z)>0.5).float()
        
        for i in range(n_samples):
            adj_matrix = fake_adj[i].numpy()
            G = nx.from_numpy_array(adj_matrix)
            degrees = torch.sum(fake_adj[i], dim=1).numpy().round(2)
            
            plt.figure(figsize=(8, 6))
            nx.draw(G, with_labels=True, node_color='lightblue', 
                   node_size=500, font_size=16, font_weight='bold')
            plt.title(f'Generated Graph {i+1}\nNode Degrees: {degrees}')
            plt.show()

generator = Generator(n_nodes=n_nodes)

generator.load_state_dict(torch.load('generator_hard.pth'))

visualize_generated_graphs(generator, n_samples=5, n_nodes=n_nodes)