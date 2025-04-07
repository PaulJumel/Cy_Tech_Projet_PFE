import numpy as np
from validation import is_correct

# Returns a canonical (flattened and binarized) representation to identify duplicates.
def hash_matrix(matrix, threshold, n_nodes):
    matrix = matrix.detach().cpu().numpy().reshape(n_nodes, n_nodes)
    matrix = (matrix > threshold).astype(int)
    upper_triangle = matrix[np.triu_indices(n_nodes, k=1)]  
    return tuple(upper_triangle)

# Evaluates the quality of a batch of generated graphs.
def evaluate_generated_graphs(fake_matrices, batch_size, threshold, n_nodes, type_graphs):
    correct_count = 0
    seen_hashes = set()

    for i in range(batch_size):
        h = hash_matrix(fake_matrices[i], threshold, n_nodes)
        seen_hashes.add(h)
        if is_correct(fake_matrices[i], type_graphs): 
            correct_count += 1

    unique_count = len(seen_hashes)
    correct_percentage = (correct_count / batch_size) * 100

    return unique_count, correct_percentage
