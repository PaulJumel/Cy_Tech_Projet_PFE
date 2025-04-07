# General parameters
n_nodes = 10                  # Number of nodes in each graph
epochs = 10000                # Number of training epochs
batch_size = 32               # Batch size used during training
learning_rate = 0.0001        # Learning rate for the optimizer
gamma = 0.9                   # Decay factor for the learning rate scheduler
threshold = 0.5               # Threshold value used for decisions
alpha = 10                    # Coefficient for penalty on K3,3 and similar subgraphs
num_rivals = 3                # Number of rival models or competitors (e.g., in adversarial settings)