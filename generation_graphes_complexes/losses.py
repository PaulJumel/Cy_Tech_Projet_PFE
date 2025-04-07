from compute_g_loss import planar_g_loss, cactus_g_loss, cycle_g_loss, arbre_g_loss
from compute_d_loss import planar_d_loss, cactus_d_loss, cycle_d_loss, tree_d_loss

# Computes the loss for the Discriminator based on the graph type
def compute_d_loss(type_graphs, discriminator, discriminators, real_matrices, fake_matrices, real_labels, fake_labels, adversarial_loss):

    if type_graphs == 1:  # Planar graph
        return planar_d_loss(discriminator, real_matrices, fake_matrices, real_labels, fake_labels, adversarial_loss)
    elif type_graphs == 2:  # Cactus graph
        return cactus_d_loss(discriminator, real_matrices, fake_matrices)
    elif type_graphs == 3:  # Cycle graph
        return cycle_d_loss(discriminators, real_matrices, fake_matrices)
    elif type_graphs in [4,5]: # Tree or Binary Tree
        return tree_d_loss(discriminator, real_matrices, fake_matrices, real_labels, fake_labels, adversarial_loss)
    else:
        raise ValueError("Type de graphe non pris en charge.")

# Computes the loss for the Generator based on the graph type
def compute_g_loss(type_graphs, discriminator, fake_matrices, adversarial_loss, batch_size, epoch, epochs, discriminators, num_rivals, real_labels, n_nodes):
    """
    Calcul de la perte pour le Générateur en fonction du type de graphe.
    """
    if type_graphs == 1:  # Planar graph
        g_loss = adversarial_loss(discriminator(fake_matrices), real_labels)
        return planar_g_loss(g_loss, fake_matrices, batch_size, epoch, epochs)
    elif type_graphs == 2:  # Cactus graph
        return cactus_g_loss(discriminator, fake_matrices)
    elif type_graphs == 3: # Cycle graph
        return cycle_g_loss(fake_matrices, discriminators, num_rivals)
    elif type_graphs in [4,5]: # Tree or Binary Tree
        g_loss = adversarial_loss(discriminator(fake_matrices), real_labels)
        return arbre_g_loss(g_loss, fake_matrices, n_nodes)
    else:
        raise ValueError("Type de graphe non pris en charge.")
