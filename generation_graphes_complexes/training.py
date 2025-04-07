import os
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from models import Generator, Discriminator
from data_and_hyperparams import epochs, n_nodes, batch_size, learning_rate, gamma, num_rivals, threshold
from losses import compute_d_loss, compute_g_loss
from evaluate_graphs import evaluate_generated_graphs

# Argument Parsing 
parser = argparse.ArgumentParser(description="Entraîner un GAN pour générer des matrices d'adjacence.")
parser.add_argument("--type_graphs", type=int, required=True, 
                    help="Type de graphes à générer : 1=planar, 2=cactus, 3=cycle, 4=arbre, 5=arbre binaire.")
args = parser.parse_args()
type_graphs = args.type_graphs

# Load Training Data 
dataset_dir = "graphs_matrices"
matrices = []

for file in sorted(os.listdir(dataset_dir)):
    if file.endswith(".txt"):
        matrix = np.loadtxt(os.path.join(dataset_dir, file), dtype=int)
        matrices.append(matrix)

train_data = np.array(matrices).astype(float)  
n_nodes = train_data.shape[1] 
print(f"Taille des données d'entraînement : {train_data.shape}")

# Initialize Models 
generator = Generator(n_nodes, type_graphs)
discriminator = Discriminator(n_nodes, type_graphs)

# Adversarial Loss Function 
adversarial_loss = nn.BCELoss()

# Labels for real and fake data
real_labels = torch.full((batch_size, 1), 0.9)  
fake_labels = torch.full((batch_size, 1), 0.1)  

# Optimizers and learning rate schedulers
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
scheduler_G = optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=gamma)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Initialize multiple discriminators (rivals) for improved diversity in training (only used by graph cycle)
discriminators = [Discriminator(n_nodes) for _ in range(num_rivals)]
optimizers_d = [torch.optim.Adam(d.parameters(), lr=learning_rate) for d in discriminators]

# Lists to track progress (for visualization)
percentage_valid_graphs = []  # Track the percentage of valid graphs
g_loss_list = []  # Track Generator loss
d_loss_list = []  # Track Discriminator loss
unique_graphs_counts = []  # Track the number of unique graphs generated

# === Training Loop ===
for epoch in range(epochs):
    idx = np.random.randint(0, train_data.shape[0], batch_size)   # Randomly sample a batch of matrices from the training data
    real_matrices = torch.tensor(train_data[idx], dtype=torch.float32)   # Convert selected matrices to tensor for processing

    z = torch.randn(batch_size, n_nodes * n_nodes)  # Random latent vectors (noise) as input for the generator
    fake_matrices = generator(z)  # Generate fake adjacency matrices from the latent vectors

    # Discriminator Training 
    should_train_d = not (type_graphs in [4, 5] and epoch % 2 != 0)

    if should_train_d : 
        optimizer_D.zero_grad()
        d_loss = compute_d_loss(type_graphs, discriminator, discriminators, real_matrices, fake_matrices, real_labels, fake_labels, adversarial_loss)
        d_loss.backward()
        optimizer_D.step()
    
    # Generator Training
    optimizer_G.zero_grad()
    g_loss = compute_g_loss(type_graphs, discriminator, fake_matrices, adversarial_loss, batch_size, epoch, epochs, discriminators, num_rivals, real_labels, n_nodes)
    g_loss.backward()
    optimizer_G.step()  
    
    if type_graphs == 1:
        scheduler_G.step()

    # Monitoring and Evaluation
    if epoch % (epochs // 100) == 0:
        unique_count, correct_percentage = evaluate_generated_graphs(fake_matrices, batch_size, threshold, n_nodes, type_graphs)

        unique_graphs_counts.append(unique_count)
        percentage_valid_graphs.append(correct_percentage)
        g_loss_list.append(g_loss.cpu().detach().numpy())
        d_loss_list.append(d_loss.cpu().detach().numpy())

        print(f"Epoch {epoch}/{epochs} - D Loss: {d_loss.item():.4f} - G Loss: {g_loss.item():.4f} - "
            f"Matrices correctes: {correct_percentage:.2f}% - Graphes uniques: {unique_count}")
    
# === Save Visualization Plots ===

# Plot the number of unique graphs generated over time
plt.figure()
plt.plot(unique_graphs_counts, color='orange')
plt.title("Nombre de graphes différents générés")
plt.xlabel("Epochs")
plt.ylabel("Graphes uniques")
plt.grid(True)
plt.savefig(f"diversity_graph_{type_graphs}.png", dpi=300, bbox_inches='tight')

# Plot the percentage of valid graphs generated over time
plt.figure()
plt.plot(percentage_valid_graphs)
plt.title("Pourcentage de graphes valides")
plt.xlabel("Epochs")
plt.ylabel("Pourcentage")
plt.grid(True)
plt.savefig(f"graph_correct_{type_graphs}.png", dpi=300, bbox_inches='tight')

# Plot the Generator and Discriminator losses over time
plt.figure()
plt.plot(g_loss_list, label="G Loss")
plt.plot(d_loss_list, label="D Loss")
plt.title("Pertes du générateur et du discriminateur")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig(f"generator_and_descriminator_{type_graphs}.png", dpi=300, bbox_inches='tight')

# === Save the Trained Generator Weights ===
torch.save(generator.state_dict(), f"generator_trained_{type_graphs}.pth")
print("Générateur entraîné sauvegardé.")
    