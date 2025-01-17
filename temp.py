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
    total_loss = torch.zeros(batch_size).to(adj_matrix.device)
    time_loss = torch.zeros(batch_size).to(adj_matrix.device)
    for i in range(batch_size):
        # Utiliser le straight-through estimator
        matrice = (adj_matrix[i] > 0.5).float() + adj_matrix[i] - adj_matrix[i].detach()
        # Convertir la matrice d'adjacence en un graphe NetworkX
        G = nx.from_numpy_array(matrice.detach().cpu().numpy())
        
        start_time = time.time()
        # Trouver les cycles dans le graphe
        maximal_cliques = nx.find_cliques(G)
        end_time = time.time()
        # Calculer la perte basée sur les arêtes qui forment les cycles
        time_loss[i] = end_time - start_time
    time_loss = -torch.sum(time_loss)
    return time_loss

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
    
    #Add a new node u_ij for each anti-edge u_i u_j in N(v)
    new_nodes = {}
    for u, w in anti_edges:
        new_node = f"{u}_{w}"
        G.add_node(new_node)
        new_nodes[(u, w)] = new_node
    
    #Add edges between each u_ij and the nodes in N(u_i) ∪ N(u_j)
    for (u, w), new_node in new_nodes.items():
        neighbors_u_w = set(G.neighbors(u)).union(set(G.neighbors(w)))
        for neighbor in neighbors_u_w:
            G.add_edge(new_node, neighbor)
    
    #Add one edge between each pair of new nodes
    new_node_list = list(new_nodes.values())
    for i in range(len(new_node_list)):
        for j in range(i + 1, len(new_node_list)):
            G.add_edge(new_node_list[i], new_node_list[j])
    
    #Remove N[v]
    G.remove_nodes_from(neighbors)
    G.remove_node(v)
    
    return G


def mis(G):
    if G.number_of_nodes() == 0:
        return 0, set()
    
    #Connected components
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
    
    #Node domination
    for v in G.nodes():
        for w in G.nodes():
            if v != w and set(G.neighbors(v)).union({v}).issubset(set(G.neighbors(w)).union({w})):
                H = G.copy()
                print(v,w)
                H.remove_node(w)
                return mis(H)
    
    #Foldable node
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
    
    #v is the node of maximum degree
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
    # Create empty adjacency matrix
    adj_matrix = np.zeros((n, n))
    
    # Fill upper triangle with random edges
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1  # Make it symmetric
                
    return adj_matrix

    
def train(n_nodes=10,
           n_graphs=256,
           num_epochs=10_000,
           batch_size=32,
           fct=generate_erdos_renyi,
           loss_algo=maximal_clique_time_loss):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    time_ratios = []
    graph_dataset = [fct(n_nodes) for _ in range(n_graphs)]
    
    generator = Generator(n_nodes).to(device)
    discriminator = Discriminator(n_nodes).to(device)
    
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0001)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)

    for epoch in range(num_epochs):
        for i in range(0, n_graphs, batch_size):
            batch_graphs_np = np.array(graph_dataset[i:i+batch_size])
            real_graphs = torch.tensor(batch_graphs_np, dtype=torch.float32).to(device)

            
            
            z = torch.randn(batch_size, n_nodes).to(device)
            
            fake_adj = generator(z)
            

            d_optimizer.zero_grad()
            d_real = discriminator(real_graphs)
            d_fake = discriminator(fake_adj.detach())
            d_loss = -torch.mean(torch.log(d_real) + torch.log(1 - d_fake))
            d_loss.backward()
            d_optimizer.step()
            
            g_optimizer.zero_grad()
            g_fake = discriminator(fake_adj)
            reference_time = loss_algo(real_graphs)
            time_algo = loss_algo(fake_adj)
            time_ratios.append((time_algo/reference_time).cpu().detach().numpy())
            g_loss = -torch.mean(torch.log(1-g_fake)) + 50*time_algo
            g_loss.backward()
            g_optimizer.step()
            
            
        if epoch % 1 == 0:
            print(f'''Epoch [{epoch}/{num_epochs}],
                d_loss: {d_loss.item()},
                g_loss: {g_loss.item()},
                time_algo: {time_algo}''')
            
            torch.save(generator.state_dict(), 'generator_cycle2.pth')
            torch.save(discriminator.state_dict(), 'discriminator_cycle.pth')
    plt.figure(figsize=(10, 6))
    plt.plot(time_ratios)
    plt.xlabel('Iteration')
    plt.ylabel('Time Ratio (time_algo/reference_time)')
    plt.title('Algorithm Time Ratio Evolution')
    plt.grid(True)
    plt.show()
n_nodes = 10
train(n_nodes=n_nodes, n_graphs=1024, num_epochs=350, batch_size=32, fct=generate_erdos_renyi)

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

generator.load_state_dict(torch.load('generator_cycle2.pth'))

visualize_generated_graphs(generator, n_samples=5, n_nodes=n_nodes)