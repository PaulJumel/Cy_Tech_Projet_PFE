#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:42:40 2025

@author: cytech
"""


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random as rd
import time
#from arbres import calculate_different_graphs, adjacency

# Fonction pour obtenir la matrice d'adjacence
def adjacency(G): # Parfois,La fonction de nx donne la même matrice à isomorphisme près.
    
    n=len(G.nodes())
    
    matrice=np.zeros((n,n))
    
    for e in list(G.edges()):
        matrice[e]=1
        matrice[e[1],e[0]]=1
    
    return matrice


def generate_dataset30(factor):
    # Liste de nombres d'arêtes
    edge_counts = [375, 374, 371, 372, 373, 370, 392, 383, 369, 368, 362, 380, 
                   388, 384, 367, 387, 386, 385, 381, 379, 391, 377, 366, 390, 
                   389, 382, 376, 364, 378, 365, 363, 361, 360, 359, 358, 393, 
                   394, 357, 355, 356, 351, 354, 353, 349, 350, 348, 352, 346, 
                   347, 342]

    graphs = []  # Liste pour stocker les graphes générés

    for nbedges in edge_counts:
        for _ in range(factor):
            graph = nx.gnm_random_graph(30, nbedges)  # Générer un graphe avec 30 sommets et num_edges arêtes
            graphs.append(graph)
    rd.shuffle(graphs)

    return graphs

def generate_dataset50(factor):
    # Liste de nombres d'arêtes
    edge_counts = [1116,1118,1119,1120,1121,1122,1123,1124,1125,1126,1127,1128,
                   1129,1130,1131,1132,1135,1136,1137,1138,1139,1140,1141,1142,
                   1143,1144,1146]

    graphs = []  # Liste pour stocker les graphes générés

    for nbedges in edge_counts:
        for _ in range(factor):
            graph = nx.gnm_random_graph(50, nbedges)  # Générer un graphe avec 50 sommets et nbedges arêtes
            graphs.append(graph)
    rd.shuffle(graphs)

    return graphs

# Fonction pour calculer le temps
def calculate_time_graphs(fake_matrices, threshold=0.5):
    time_graphs = 0
    for fake_matrix in fake_matrices:
        binary_matrix = (fake_matrix > threshold).astype(int)
        G = nx.from_numpy_array(binary_matrix)
        d=time.time()
        max(nx.find_cliques(G), key=len)
        f=time.time()
        time_graphs+=(f-d)
    return time_graphs / len(fake_matrices)

def calculate_edges_graphs(fake_matrices, threshold=0.5):
    edges_graphs = 0
    for fake_matrix in fake_matrices:
        binary_matrix = (fake_matrix > threshold).astype(int)
        G = nx.from_numpy_array(binary_matrix)
        edges_graphs+=( G.number_of_edges() )
    return edges_graphs / len(fake_matrices)

def calculate_different_graphs(fake_matrices, threshold=0.5):
    unique_graphs = set()
    for fake_matrix in fake_matrices:
        binary_matrix = (fake_matrix > threshold).astype(int)
        #G = nx.from_numpy_array(binary_matrix)
        unique_graphs.add(binary_matrix.tostring())  # Représentation unique
    return len(unique_graphs) / len(fake_matrices) * 100

# Le générateur (essaye de tromper le discriminateur)
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

def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true * y_pred)  # y_true est 1 ou -1


def interpolate_and_generate(generator, z1, z2, steps):
    interpolated_graphs = []
    for alpha in torch.linspace(0, 1, steps):
        z = z1 * (1 - alpha) + z2 * alpha  # Interpolation linéaire
        generated_graph = generator(z.unsqueeze(0))
        interpolated_graphs.append(generated_graph.detach())
    return torch.cat(interpolated_graphs)
def diversity_metric(graph):
    """
    Mesure une métrique de diversité (à personnaliser).
    Exemple : Nombre d'arêtes, variance des connexions.
    """
    adj_matrix = graph.squeeze().numpy()
    unique_edges = np.unique(adj_matrix > 0.5)  # Seulement les arêtes
    return len(unique_edges) / (adj_matrix.shape[0] ** 2)

#
def resample_generated_graphs(graphs):
    """
    Ré-échantillonne les graphes pour encourager la diversité.
    :param graphs: Tensor des graphes générés (batch_size, n_nodes, n_nodes)
    :return: Tensor des graphes ré-échantillonnés
    """
    resampled = []
    for graph in graphs:
        # Appliquer une perturbation sur chaque graphe
        perturbed_graph = graph + 0.1 * torch.randn_like(graph)
        perturbed_graph = torch.clamp(perturbed_graph, 0, 1)  # Normaliser entre 0 et 1
        
        # Ajouter un critère de diversité (par ex., conserver des graphes avec des cycles spécifiques)
        if torch.sum(perturbed_graph) > 0.5 * graph.numel():  # Critère arbitraire
            resampled.append(perturbed_graph)
        else:
            resampled.append(graph)
    
    return torch.stack(resampled)

def generate_conditional_noise(batch_size, n_nodes, condition_dim=10):
    """
    Génère un bruit conditionné en fonction de conditions spécifiques.
    :param batch_size: Taille du lot.
    :param n_nodes: Nombre de nœuds dans le graphe.
    :param condition_dim: Dimension de l'information conditionnelle.
    :return: Tensor de bruit conditionné.
    """
    noise = torch.randn(batch_size, n_nodes * n_nodes)  # Bruit aléatoire
    conditions = torch.randn(batch_size, condition_dim)  # Conditions aléatoires (ou basées sur des données spécifiques)
    
    # Combiner le bruit et les conditions
    conditional_noise = torch.cat((noise, conditions), dim=1)
    return conditional_noise

def clique_solver_time_loss(graphs, device="cpu"):
    """
    Calcule une perte qui favorise les graphes prenant le plus de temps 
    à résoudre avec l'algorithme de détection de clique maximale de NetworkX.

    Args:
        graphs (torch.Tensor): Un batch de matrices d'adjacence de graphes (batch_size, n_nodes, n_nodes).
        device (str): L'appareil sur lequel exécuter la perte (cpu/gpu).

    Returns:
        torch.Tensor: La valeur de la perte, à minimiser.
    """
    batch_size = graphs.shape[0]
    times = []

    for i in range(batch_size):
        adj_matrix = graphs[i].detach().cpu().numpy()  # Convertir en numpy
        G = nx.from_numpy_array((adj_matrix > 0.5).astype(int))  # Binarisation du graphe

        start_time = time.time()
        _ = max(nx.find_cliques(G), key=len)  # Exécuter l'algorithme
        elapsed_time = time.time() - start_time

        times.append(elapsed_time)

    times_tensor = torch.tensor(times, dtype=torch.float32, device=device)

    # Inverser le temps pour que les graphes plus lents aient une plus petite perte
    loss = -torch.mean(times_tensor)  # On maximise le temps en minimisant la perte

    return loss

def number_of_edges_loss(graphs, target_min=1116, target_max=1146, device="cpu"):
    """
    Calcule une perte qui pénalise les graphes ayant un nombre d’arêtes en dehors de l’intervalle [target_min, target_max].

    Args:
        graphs (torch.Tensor): Batch de matrices d'adjacence (batch_size, n_nodes, n_nodes).
        target_min (int): Borne inférieure du nombre d’arêtes souhaité.
        target_max (int): Borne supérieure du nombre d’arêtes souhaité.
        device (str): CPU ou GPU.

    Returns:
        torch.Tensor: Valeur de la perte.
    """
    batch_size = graphs.shape[0]
    
    # Calculer le nombre d’arêtes pour chaque graphe
    edge_counts = graphs.sum(dim=(1, 2)) / 2  # Division par 2 pour éviter le double comptage des arêtes
    
    # Calculer la distance à l’intervalle cible
    lower_diff = torch.clamp(target_min - edge_counts, min=0)  # Pénalité si trop peu d’arêtes
    upper_diff = torch.clamp(edge_counts - target_max, min=0)  # Pénalité si trop d’arêtes
    
    # Perte basée sur l’éloignement de l’intervalle
    loss = lower_diff + upper_diff*40
    return loss.mean().to(device)


def train(n_nodes=30, num_epochs=1_000, batch_size=32):
    """
    Entraîne un GAN pour générer des graphes.

    Args:
        n_nodes (int): Nombre de sommets, 30 par défaut.
        num_epochs (int): Nombre d'époques pour l'entraînement, 1_000 par défaut.
        batch_size (int): Taille des lots pour l'entraînement, 32 par défaut.
    """
    
    # Créer les modèles
    generator = Generator(n_nodes)
    discriminator = Discriminator(n_nodes)

    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    #scheduler_G = optim.lr_scheduler.ExponentialLR(g_optimizer, gamma=0.9)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    

    loss_fn = nn.BCELoss()
    #loss_fn  = wasserstein_loss
    
    edges_loss = 0
    
    if n_nodes==30:
        generate_dataset=generate_dataset30
    elif n_nodes==50:
        generate_dataset=generate_dataset50
    else:
        raise ValueError("Pour l'instant, la fonction ne prend en charge que que les noeuds de 30 à 50")
        
    
    graph_dataset = np.array([adjacency(g) for g in generate_dataset(factor=32)])
    n_graphs=len(graph_dataset)
    rd.shuffle(graph_dataset)
    graph_dataset = graph_dataset.reshape(n_graphs, -1)
    # Enregistrer les graphes dans le dataset
    #save_graphs(graph_dataset, n_graphs, n_nodes)

    d_losses = []
    g_losses = []
    
    different_graph_percentages = []
    time_graph_mean = []
    edges_graphs_mean = []
    
    # Boucle d'entraînement principale
    for epoch in range(num_epochs):
        # Étape 1: Entraîner le Discriminateur avec des graphes réels et générés
        '''
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        real_labels = torch.full((batch_size, 1), 0.9)  
        fake_labels = torch.zeros(batch_size, 1)
        '''
        #scheduler_G.step()
        
        real_labels = torch.clamp(torch.full((batch_size, 1), 0.9) + 0.05 * torch.randn(batch_size, 1), 0, 1)# Smoothing des labels réels et ajout d'un bruit aléatoire sur les labels pour augmenter la robustesse du modèle
        fake_labels = torch.clamp(0.05 * torch.randn(batch_size, 1), 0, 1)



        
        if epoch%2==0:
            d_optimizer.zero_grad()
            
            z = torch.randn(batch_size, n_nodes) # Bruit aléatoire pour le générateur
            fake_graphs = generator(z)
            
            
            idx = np.random.randint(0, n_graphs, batch_size)
            real_graphs = torch.tensor(graph_dataset[idx], dtype=torch.float32)
            real_loss = loss_fn(discriminator(real_graphs), real_labels)
            fake_loss = loss_fn(discriminator(fake_graphs.detach()), fake_labels)
            
            d_loss = (real_loss + fake_loss)/2
            d_loss.backward()
            d_optimizer.step()
        

        # Étape 2: Entraîner le Générateur pour tromper le Discriminateur
        g_optimizer.zero_grad()
        z = torch.randn(batch_size, n_nodes)  # Bruit aléatoire pour le générateur
        fake_graphs = generator(z)
        
        # *** Ré-échantillonnage ***
        sampled_graphs = resample_generated_graphs(fake_graphs)
        
        generated_loss = loss_fn(discriminator(sampled_graphs), torch.ones(batch_size, 1))  # Le générateur essaie de tromper le discriminateur
        
        
        if epoch>10500: #N'est appliqué le loss que lorsque le nombre d'epoch est élevé, que l'apprentissage est bien entammé (le nombre d'arrêtes avant étant trop petit)
            edges_loss = number_of_edges_loss(sampled_graphs)
        
        total_loss = 0.1*generated_loss + 0.05*edges_loss
        total_loss.backward()
        g_optimizer.step()
        
        d_losses.append(d_loss.item())
        g_losses.append(total_loss.item())
        
        
    
        # Stocker les métriques
        different_graph_percentages.append(calculate_different_graphs(fake_graphs.detach().numpy()))
        if epoch%10==0: #Réduit le temps d'apprentissage (Cette métrique n'intervient pas dans l'apprentissage)
            time_graph_mean.append(calculate_time_graphs(fake_graphs.detach().numpy()))
        edges_graphs_mean.append(calculate_edges_graphs(fake_graphs.detach().numpy()))
        

        # Afficher la loss toutes les 500 époques
        if epoch % 100 == 0:
            threshold = 0.5

            print(f"[Epoch {epoch}/{num_epochs}] D loss: {d_loss.item():.4f} | "
                  f"G loss: {generated_loss.item():.4f} | ")

            # Affichage des graphes
            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            sample_indices = np.random.choice(fake_graphs.shape[0], 5, replace=False)
            for i, ax in zip(sample_indices, axes):
                binary_matrix = (fake_graphs[i].detach().numpy() > threshold).astype(int)
                G = nx.from_numpy_array(binary_matrix)
                nx.draw(G, ax=ax, node_size=20, font_size=8)
                ax.set_title(f"Graph {i}")
            plt.show()
                
                
                
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Loss du discriminator')
    plt.plot(g_losses, label='Loss du generator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss du GAN')
    plt.show()
    
    # Plot du pourcentage de graphes différents
    plt.subplot(2, 2, 3)
    plt.plot(different_graph_percentages, label="Different Graphs (%)")
    plt.title("Different Graphs")
    plt.xlabel("Epochs")
    plt.ylabel("Percentage")
    plt.legend()
    plt.show()
    
    # Plot du pourcentage de graphes différents
    plt.figure(figsize=(10, 5))
    plt.plot(time_graph_mean)
    plt.title("Temps de l'algo")
    plt.xlabel("Epochs")
    plt.ylabel("Temps pris")
    plt.legend()
    plt.show()
    
    # Plot du pourcentage de graphes différents
    plt.figure(figsize=(10, 5))
    plt.plot(edges_graphs_mean)
    plt.title("Nombre d'arrêtes")
    plt.xlabel("Epochs")
    plt.ylabel("Nombre d'arrêtes")
    plt.legend()
    plt.show()
    
d = time.time()
train(n_nodes=30, num_epochs=1_500, batch_size=32)
f = time.time()
duree = f - d
print("\n    Temps : ", duree)

'''
def filter_and_resample(graphs, generator, latent_dim, threshold=0.9):
    unique_graphs = []
    z_replacement = torch.randn(len(graphs), latent_dim)
    
    for idx, g in enumerate(graphs):
        if is_duplicate(g, unique_graphs, threshold):
            new_graph = generator(z_replacement[idx].unsqueeze(0)).detach()
            unique_graphs.append(new_graph)
        else:
            unique_graphs.append(g)
    return torch.stack(unique_graphs)

def is_duplicate(graph, graph_list, threshold):
    for g in graph_list:
        if torch.norm(graph - g) < threshold:
            return True
    return False
'''
