from data_and_hyperparams import threshold
import matplotlib.pyplot as plt
import networkx as nx

# Function to check if a graph is planar
def is_planar(matrix):
    adj_matrix = (matrix.detach().numpy() > threshold).astype(int)  
    G = nx.from_numpy_array(adj_matrix)
    return nx.check_planarity(G)[0]


# Function to check if a graph is a cactus
def is_cactus(matrix):
    adj_matrix = (matrix.detach().numpy()).astype(int)
    G = nx.from_numpy_array(adj_matrix)

    if nx.number_connected_components(G) > 1:
        return False
    cycles = nx.cycle_basis(G)
    edge_count = {}
    for cycle in cycles:
        for i in range(len(cycle)):
            edge = tuple(sorted((cycle[i], cycle[(i + 1) % len(cycle)])))
            if edge in edge_count:
                return False
            edge_count[edge] = 1
    return True 

# Function to check if a graph is a cycle
def is_cycle(matrix):
    adj_matrix = (matrix.detach().numpy() > threshold).astype(int) 
    G = nx.from_numpy_array(adj_matrix)
    return nx.is_connected(G) and all(deg == 2 for _, deg in G.degree())

# Function to check if a graph is a tree
def is_tree(matrix):
    adj_matrix = (matrix.detach().numpy()  > threshold).astype(int)
    G = nx.from_numpy_array(adj_matrix)

    is_connected = nx.is_connected(G)
    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()
    is_tree = is_connected and num_edges == num_nodes - 1 

    return is_tree

# Function to check if a graph is a binary tree
def is_binary_tree(matrix):
    adj_matrix = (matrix.detach().numpy() > threshold).astype(int)
    G = nx.from_numpy_array(adj_matrix)

    is_tree_check = is_tree(matrix)  
    is_binary = all(deg <= 3 for _, deg in G.degree()) 

    return is_tree_check and is_binary

# Main function to check if a generated graph matches the specified type
def is_correct(matrix, type_graphs):
    if type_graphs == 1: 
        return is_planar(matrix)
    elif type_graphs == 2:  
        return is_cactus(matrix)
    elif type_graphs == 3:
        return is_cycle(matrix)
    elif type_graphs == 4: 
        return is_tree(matrix)
    elif type_graphs == 5:
        return is_binary_tree(matrix)
    else:
        raise ValueError("Type de graphe non pris en charge.")      