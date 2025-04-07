import time
import random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

##############################
#  Helper functions
##############################
def upper_vector_to_adj(vec, n_nodes):
    # vec: shape (..., E) with E = n_nodes*(n_nodes-1)//2
    # Return a symmetric adjacency matrix with zeros on diagonal.
    device = vec.device
    # get indices for upper triangle
    rows, cols = torch.triu_indices(n_nodes, n_nodes, offset=1)
    A = torch.zeros(vec.shape[:-1] + (n_nodes, n_nodes), device=device)
    A[..., rows, cols] = vec
    A = A + A.transpose(-2, -1)
    return A

def st_round(x, threshold=0.5):
    # Straight-through rounding: forward passes hard binary, gradients flow from x.
    y = (x > threshold).float()
    return y + (x - x.detach())

def graph_to_vector(A):
    # Convert a full numpy adjacency matrix (n x n) to a binary vector from its upper triangle
    n = A.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    return A[rows, cols]

def compute_graph_features(A):
    # Given a full binary numpy adjacency matrix, compute simple features.
    G = nx.from_numpy_array(A)
    density = nx.density(G)
    clustering = np.mean(list(nx.clustering(G).values()))
    # You can add more features if needed.
    return np.array([density, clustering])

def create_graph_dataset_vector_cond(n_nodes, n_graphs):
    dataset = []
    conditions = []
    for i in range(n_graphs):
        # Use half Erdős–Rényi and half Barabási–Albert
        if i < n_graphs // 2:
            A = nx.to_numpy_array(nx.erdos_renyi_graph(n_nodes, 0.94))
        else:
            A = nx.to_numpy_array(nx.barabasi_albert_graph(n_nodes, max(1, n_nodes//10)))
        # Threshold to get binary matrix and symmetrize
        A = (A > 0.5).astype(float)
        A = np.triu(A, 1)
        A = A + A.T
        edge_vec = graph_to_vector(A)
        cond = compute_graph_features(A)  # condition has 2 features: density and average clustering
        dataset.append(edge_vec)
        conditions.append(cond)
    return np.array(dataset), np.array(conditions)

def surrogate_clique_time_loss(graphs, n_nodes):
    """
    Compute a differentiable surrogate for clique time based on the spectral radius
    of the graph’s adjacency matrix.
    """
    A = upper_vector_to_adj(graphs, n_nodes)
    # Remove self-loops.
    eye = torch.eye(n_nodes, device=A.device).unsqueeze(0)
    A = A * (1 - eye)
    # Compute eigenvalues of the symmetric matrix.
    # torch.linalg.eigvalsh returns sorted eigenvalues in ascending order.
    eigenvalues = torch.linalg.eigvalsh(A)
    max_eigen = eigenvalues[:, -1]
    # We want to encourage harder graphs, so we want to maximize the spectral radius.
    # Return loss = -mean(spectral_radius)
    return -torch.mean(max_eigen)

##############################
# Conditional GAN Models
##############################
cond_dim = 2  # our condition: [density, avg_clustering]

class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, cond_dim, num_edges):
        super(ConditionalGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.num_edges = num_edges
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_edges)
        )
    def forward(self, z, cond):
        # z: (batch, latent_dim), cond: (batch, cond_dim)
        x = torch.cat([z, cond], dim=1)
        logits = self.fc(x)
        p = torch.sigmoid(logits)
        # Use straight-through estimator for binary output
        out = st_round(p)
        return out  # shape: (batch_size, num_edges)

class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_edges, cond_dim):
        super(ConditionalDiscriminator, self).__init__()
        self.num_edges = num_edges
        self.cond_dim = cond_dim
        self.fc = nn.Sequential(
            nn.Linear(num_edges + cond_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, edge_vec, cond):
        x = torch.cat([edge_vec, cond], dim=1)
        return self.fc(x)

##############################
# Training settings
##############################
n_nodes = 60                      # fixed number of nodes per graph
n_dataset_graphs = 200            # number of graphs in dataset
latent_dim = 20
num_epochs = 50
batch_size = 64
num_edges = n_nodes * (n_nodes - 1) // 2

# Create dataset (edge vectors and condition vectors)
dataset_np, cond_np = create_graph_dataset_vector_cond(n_nodes, n_dataset_graphs)
dataset = torch.tensor(dataset_np, dtype=torch.float32)
dataset_cond = torch.tensor(cond_np, dtype=torch.float32)

# Compute baseline: clique times for dataset graphs
dataset_times = []
for vec in dataset_np:
    vec_tensor = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
    A_full = upper_vector_to_adj(vec_tensor, n_nodes)[0].cpu().numpy()
    A = nx.from_numpy_array(A_full)
    start = time.time()
    list(nx.find_cliques(A))
    dataset_times.append(time.time() - start)
baseline_max_time = max(dataset_times)
baseline_avg_time = np.mean(dataset_times)
baseline_target = np.percentile(dataset_times, 90)  # use 90th percentile as target
print("Dataset maximum clique time: {:.4f}s".format(baseline_max_time))
print("Baseline target (90th percentile): {:.4f}s".format(baseline_target))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
baseline_target_tensor = torch.tensor(baseline_target, device=device)
# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = ConditionalGenerator(latent_dim, cond_dim, num_edges).to(device)
discriminator = ConditionalDiscriminator(num_edges, cond_dim).to(device)

g_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)

# Lists to store losses and generated clique times
g_losses = []
d_losses = []
gen_times = []

##############################
# Training Loop
##############################
early_stop = False
prev_max_sample_time = None
streak_count = 0
gen_avg_times = []
gen_max_times = []
for epoch in range(num_epochs):
    perm = torch.randperm(len(dataset))
    epoch_g_loss = 0.
    epoch_d_loss = 0.
    epoch_time_loss = 0.
    n_batches = 0
    for i in range(0, len(dataset), batch_size):
        n_batches += 1
        current_batch = dataset[perm[i:i+batch_size]].to(device)
        current_cond = dataset_cond[perm[i:i+batch_size]].to(device)
        cur_bs = current_batch.size(0)
        
        # ----- Discriminator step -----
        d_optimizer.zero_grad()
        d_real = discriminator(current_batch, current_cond)
        real_loss = -torch.log(d_real + 1e-8).mean()
        # Sample latent noise and also sample condition vectors from dataset (or use current_cond)
        z = torch.randn(cur_bs, latent_dim).to(device)
        # For generated samples, we sample a random condition from dataset_cond
        idx = torch.randint(0, dataset_cond.size(0), (cur_bs,))
        fake_cond = dataset_cond[idx].to(device)
        fake_vec = generator(z, fake_cond)
        d_fake = discriminator(fake_vec.detach(), fake_cond)
        fake_loss = -torch.log(1 - d_fake + 1e-8).mean()
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()
        
                # ----- Generator step -----
        g_optimizer.zero_grad()
        fake_vec = generator(z, fake_cond)
        d_fake = discriminator(fake_vec, fake_cond)
        g_adv_loss = -torch.log(d_fake + 1e-8).mean()
        
        # Differentiable surrogate for clique time loss.
        surrogate_loss = surrogate_clique_time_loss(fake_vec, n_nodes)
        # Additionally, encourage denser graphs.
        A_fake = upper_vector_to_adj(fake_vec, n_nodes)
        eye = torch.eye(n_nodes, device=device).unsqueeze(0)
        A_fake = A_fake * (1 - eye)  # remove self–loops
        density = A_fake.sum(dim=(1,2)) / (n_nodes * (n_nodes-1)/2)
        density_target = 0.6
        density_loss = torch.relu(density_target - density).mean()
        
        # Increase weight on the surrogate loss if needed.
        lambda_surrogate = 1000.0
        lambda_density = 100.0
        
        g_loss = g_adv_loss + lambda_surrogate * surrogate_loss + lambda_density * density_loss
        
        g_loss.backward()
        g_optimizer.step()
        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()
        epoch_time_loss += surrogate_loss.item()
    avg_g = epoch_g_loss / n_batches
    avg_d = epoch_d_loss / n_batches
    avg_time = - (epoch_time_loss / n_batches)
    g_losses.append(avg_g)
    d_losses.append(avg_d)
    gen_times.append(avg_time)
    if epoch % 1 == 0:
        sample_times_eval = []
        # Evaluate on 50 generated graphs (or any sample size you choose)
        generator.eval()
        with torch.no_grad():
            for _ in range(20):
                z_eval = torch.randn(1, latent_dim).to(device)
                idx_eval = torch.randint(0, dataset_cond.size(0), (1,))
                cond_eval = dataset_cond[idx_eval].to(device)
                fake_vec_eval = generator(z_eval, cond_eval)
                A_eval = upper_vector_to_adj(fake_vec_eval, n_nodes)
                A_binary = (A_eval[0] > 0.5).float().cpu().numpy()
                G_eval = nx.from_numpy_array(A_binary)
                start_eval = time.time()
                list(nx.find_cliques(G_eval))
                sample_times_eval.append(time.time() - start_eval)
        generator.train()
        avg_sample_time = np.mean(sample_times_eval)
        max_sample_time = np.max(sample_times_eval)
        gen_max_times.append(max_sample_time)
        gen_avg_times.append(avg_sample_time)
        print("Epoch {} Evaluation: Avg Gen Clique Time {:.4f}s, Max Gen Clique Time {:.4f}s".format(
            epoch, avg_sample_time, max_sample_time))
        if max_sample_time > max(gen_max_times):
            best_max_time = max_sample_time
            best_epoch = epoch
            torch.save(generator.state_dict(), "best_generator_max_gen.pth")
            print("New best model saved at epoch {} with max_gen_clique_time {:.4f}s".format(epoch, max_sample_time))
        # Early stopping callback: if max_sample_time increases twice in a row, stop training.
        if prev_max_sample_time is None:
            prev_max_sample_time = max_sample_time
        else:
            if max_sample_time < prev_max_sample_time:
                streak_count += 1
            else:
                streak_count = 0
            prev_max_sample_time = max_sample_time
            if streak_count >= 2:
                print("Early stopping triggered at epoch {} (max sample time raised twice in a row).".format(epoch))
                early_stop = True
        if early_stop:
            break

##############################
# Evaluation & Plots
##############################
# Evaluate average clique time on generated graphs using 50 samples.
sample_times = []
gen_features = []  # collect generated condition features for analysis
generator.eval()
with torch.no_grad():
    for _ in range(50):
        z = torch.randn(1, latent_dim).to(device)
        # For evaluation, sample a random condition from dataset_cond
        idx = torch.randint(0, dataset_cond.size(0), (1,))
        cond_sample = dataset_cond[idx].to(device)
        fake_vec = generator(z, cond_sample)
        A = upper_vector_to_adj(fake_vec, n_nodes)
        A_bin = (A[0] > 0.5).float().cpu().numpy()
        G_sample = nx.from_numpy_array(A_bin)
        start = time.time()
        list(nx.find_cliques(G_sample))
        sample_times.append(time.time() - start)
        # Also compute generated graph features
        features = compute_graph_features(A_bin)
        gen_features.append(features)
avg_gen_time = np.mean(sample_times)
max_gen_time = np.max(sample_times)
print("Average clique time on generated graphs: {:.4f}s".format(avg_gen_time))
print("Maximum clique time on generated graphs: {:.4f}s".format(max_gen_time))
print("Maximum clique time in dataset: {:.4f}s".format(baseline_max_time))

# Plot losses and generated clique times evolution
plt.figure(figsize=(14,6))
plt.subplot(1,3,1)
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()

plt.subplot(1,3,2)
plt.plot(gen_max_times, label='Generated Max Clique Time')
plt.plot([baseline_max_time]*len(gen_times), 'r--', label='Dataset Max clique time')
plt.xlabel('Epoch')
plt.ylabel('Time (s)')
plt.title('Clique Finding Max Times')
plt.legend()

plt.subplot(1,3,3)
plt.plot(gen_avg_times, label='Generated Average Clique Time')
plt.plot([baseline_avg_time]*len(gen_times), 'r--', label='Dataset Average clique time')
plt.xlabel('Epoch')
plt.ylabel('Time (s)')
plt.title('Clique Finding Average Times')
plt.legend()

plt.tight_layout()
plt.show()
# Plot condition features distribution (density and clustering)
gen_features = np.array(gen_features)
dataset_features = dataset_cond.cpu().numpy() if dataset_cond.device.type=='cpu' else dataset_cond.cpu().numpy()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.hist(gen_features[:,0], bins=10, alpha=1, label='Generated Density')
plt.xlabel('Density')
plt.ylabel('Number of graphs')
plt.legend()
plt.title('Density Distribution')

plt.subplot(1,2,2)
plt.hist(dataset_features[:,0], bins=10, alpha=1, label='Dataset Density')
plt.xlabel('Density')
plt.ylabel('Number of graphs')
plt.legend()
plt.title('Density Distribution')

plt.tight_layout()
plt.show()
"""
# Plot condition features distribution (density and clustering)
gen_features = np.array(gen_features)
dataset_features = dataset_cond.cpu().numpy() if dataset_cond.device.type=='cpu' else dataset_cond.cpu().numpy()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.hist(dataset_features[:,0], bins=20, alpha=0.5, label='Dataset Density')
plt.hist(gen_features[:,0], bins=20, alpha=0.5, label='Generated Density')
plt.xlabel('Density')
plt.ylabel('Frequency')
plt.legend()
plt.title('Density Distribution')

plt.subplot(1,2,2)
plt.hist(dataset_features[:,1], bins=20, alpha=0.5, label='Dataset Avg Clustering')
plt.hist(gen_features[:,1], bins=20, alpha=0.5, label='Generated Avg Clustering')
plt.xlabel('Average Clustering Coefficient')
plt.ylabel('Frequency')
plt.legend()
plt.title('Clustering Distribution')
plt.tight_layout()
plt.show()"""