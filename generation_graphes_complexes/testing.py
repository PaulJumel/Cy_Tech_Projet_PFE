import torch
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from models import Generator
from data_and_hyperparams import n_nodes, threshold, batch_size

# === Argument parsing ===
parser = argparse.ArgumentParser(description="Entraîner un GAN pour générer des matrices d'adjacence.")
parser.add_argument("--type_graphs", type=int, required=True, 
                    help="Type de graphes à générer : 1=planar, 2=cactus, 3=cycle, 4=arbre, 5=arbre binaire.")
args = parser.parse_args()
type_graphs = args.type_graphs

# Define the model filename based on the graph type (specific trained generator model for each graph type)
model_file = f"generator_trained_{type_graphs}.pth"

# === Load the trained generator model ===
generator = Generator(n_nodes, type_graphs) 
generator.load_state_dict(torch.load(model_file)) 
generator.eval() 
print(f"Générateur entraîné chargé depuis {model_file}.")

# === Generate graphs ===
z = torch.randn(batch_size, n_nodes * n_nodes) 
fake_matrices = generator(z).detach().numpy()

# === Display the generated graphs ===
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, ax in enumerate(axes):
    binary_matrix = (fake_matrices[i] > threshold).astype(int) 
    G = nx.from_numpy_array(binary_matrix)
    nx.draw(G, ax=ax, node_size=20, font_size=8)
    ax.set_title(f"Graph {i}")
plt.show()