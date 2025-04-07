import networkx as nx
import torch
import numpy as np

# === Penalty and g_loss calculation for planar graphs ===

# Check if the graph contains forbidden subgraphs K3,3 or K5
def contains_subgraph_k33_k5(G):
    K33 = nx.complete_bipartite_graph(3, 3)
    K5 = nx.complete_graph(5)
    matcher_k33 = nx.algorithms.isomorphism.GraphMatcher(G, K33)
    matcher_k5 = nx.algorithms.isomorphism.GraphMatcher(G, K5)
    return matcher_k33.subgraph_is_isomorphic() or matcher_k5.subgraph_is_isomorphic()

# Penalty function for the presence of K3,3 or K5 subgraphs
def penalty_k33_k5(fake_matrices, threshold=0.5):
    penalty = 0
    for fake_matrix in fake_matrices:
        binary_matrix = (fake_matrix > threshold).astype(int)
        G = nx.from_numpy_array(binary_matrix)
        if contains_subgraph_k33_k5(G):
            penalty += 1
    return penalty

# Penalty function for graph density being too high
def penalty_density(fake_matrices, density_threshold=0.5, threshold=0.5):
    penalty = 0
    for fake_matrix in fake_matrices:
        binary_matrix = (fake_matrix > threshold).astype(int)
        G = nx.from_numpy_array(binary_matrix)
        density = nx.density(G)
        if density > density_threshold:
            penalty += (density - density_threshold)
    return penalty

# Generator loss for planar graphs including structural penalties
def planar_g_loss(g_loss, fake_matrices, batch_size, epoch, epochs):
    penalty_k33_k5_val = penalty_k33_k5(fake_matrices.detach().numpy())
    penalty_density_val = penalty_density(fake_matrices.detach().numpy(), density_threshold=0.5)

    g_loss = (
        g_loss
        + 10 * penalty_k33_k5_val / batch_size
        + max(1 - (epoch / epochs), 0.1) * 5 * penalty_density_val / batch_size
    )

    return g_loss

# === Generator loss for cactus graphs ===

def cactus_g_loss(discriminator, fake_graphs):
    """Calcule la perte pour le Générateur."""
    g_fake = discriminator(fake_graphs)
    g_loss = -torch.mean(torch.log(g_fake)) # Encourage generator to produce outputs that fool the discriminator
    return g_loss

# === Penalty and g_loss calculation for cyclic graphs ===

# Penalize graphs that are not connected
def connectivity_loss(adj_matrix):
    batch_size, n_nodes, _ = adj_matrix.shape
    loss = 0.0

    for i in range(batch_size):
        graph = nx.from_numpy_array(adj_matrix[i].detach().cpu().numpy())
        if not nx.is_connected(graph):
            loss += 1.0 

    return torch.tensor(loss, requires_grad=True, device=adj_matrix.device)

# Encourage diversity among graphs in the batch
def diversity_loss(graph_batch):
    if isinstance(graph_batch, np.ndarray):
        graph_batch = torch.tensor(graph_batch, dtype=torch.float32)

    diversity = 0
    batch_size = len(graph_batch)
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            diversity += torch.norm(graph_batch[i] - graph_batch[j]) 
    return -diversity / (batch_size * (batch_size - 1))

# Total generator loss for cyclic graphs
def cycle_g_loss(fake_adj, discriminators, num_rivals):
    g_fake_outputs = [d(fake_adj) for d in discriminators]

    # Rivalry loss: generator tries to fool all discriminators
    g_rivalry_loss = sum(-torch.mean(torch.log(1 - g_fake)) for g_fake in g_fake_outputs) / num_rivals

    # Penalties: degree, connectivity, and diversity
    degrees = torch.sum(fake_adj, dim=2)
    degree_penalty = torch.sum(torch.sum((degrees - 2.0) ** 2, dim=1), dim=0)
    connectivity_penalty = connectivity_loss(fake_adj)
    differents_graphs_loss = diversity_loss(fake_adj)

    # Total generator loss
    g_loss = g_rivalry_loss + degree_penalty*0.05 + connectivity_penalty*1 + differents_graphs_loss
    return g_loss

# === Penalty and g_loss calculation for trees and binary trees ===

# Penalize difference from expected number of edges in a tree (n_nodes - 1)
def edge_count_loss(adj_matrix, n_nodes):
    edge_counts = adj_matrix.sum(dim=(1, 2)) / 2
    target_edges = n_nodes - 1
    return ((edge_counts - target_edges) ** 2).mean()

# Penalize graphs with isolated nodes (degree = 0)
def degree_loss(fake_graphs):
    batch_size, n_nodes, _ = fake_graphs.size()
    loss = 0.

    for adj in fake_graphs:
        adj_bin = (adj.detach().cpu().numpy() > 0.5).astype(int)
        G = nx.from_numpy_array(adj_bin)
        if 0 in G.degree:
            loss+=1
    return loss/batch_size

# Total generator loss for tree-structured graphs
def arbre_g_loss(g_loss, fake_graphs, n_nodes):
    g_loss = 0.1*g_loss + 1.7*degree_loss(fake_graphs)+ 0.7 * diversity_loss(fake_graphs)  +0.18 * edge_count_loss(fake_graphs, n_nodes)
    return g_loss