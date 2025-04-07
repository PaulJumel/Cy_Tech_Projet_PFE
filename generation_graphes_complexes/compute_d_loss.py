import torch

# Function to compute the loss for a "planar" discriminator
def planar_d_loss(discriminator, real_matrices, fake_matrices, real_labels, fake_labels, loss_fn):
    real_loss = loss_fn(discriminator(real_matrices), real_labels)
    fake_loss = loss_fn(discriminator(fake_matrices.detach()), fake_labels)
    return (real_loss + fake_loss) / 2

# Function to compute the loss for a "cactus" discriminator
def cactus_d_loss(discriminator, real_graphs, fake_graphs):
    """Calcule la perte pour le Discriminateur."""
    d_real = discriminator(real_graphs)
    d_fake = discriminator(fake_graphs.detach())
    loss = -torch.mean(torch.log(d_real) + torch.log(1 - d_fake))
    return loss

# Function to compute the cycle loss for multiple discriminators
def cycle_d_loss(discriminators, real_matrices, fake_matrices):
    total_loss = 0
    for d, real_matrix, fake_matrix in zip(discriminators, real_matrices, fake_matrices):
        d_real = d(real_matrix)
        d_fake = d(fake_matrix.detach())
        loss_real = -torch.mean(torch.log(d_real))
        loss_fake = -torch.mean(torch.log(1 - d_fake))
        total_loss += loss_real + loss_fake

    return total_loss

# Function to compute the loss for an "tree" discriminator
def tree_d_loss(discriminator, real_matrices, fake_matrices, real_labels, fake_labels, loss_fn):
    real_loss = loss_fn(discriminator(real_matrices), real_labels)
    fake_loss = loss_fn(discriminator(fake_matrices.detach()), fake_labels)
    return (real_loss + fake_loss) / 2