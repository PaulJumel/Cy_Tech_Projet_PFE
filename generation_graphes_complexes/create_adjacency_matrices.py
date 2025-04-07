import os
import numpy as np
import networkx as nx
import random
import argparse

# === Functions to generate planar graphs ===

# Check if the graph is planar and get its planar embedding if it is
def make_graph_planar(G):
    is_planar, planar_embedding = nx.check_planarity(G)
    if is_planar:
        return G, planar_embedding

    # If the graph is not planar, iteratively remove edges until it is planar
    while not is_planar:
        edge_to_remove = random.choice(list(G.edges))
        G.remove_edge(*edge_to_remove)
        is_planar, planar_embedding = nx.check_planarity(G)

    return G, planar_embedding

#Function to generate a random planar graph ===
def generate_planar_graph(n_nodes, edge_prob):
    G = nx.gnp_random_graph(n_nodes, edge_prob)
    G_planar, planar_embedding = make_graph_planar(G)
    return G_planar

# === Functions to generate cactus graphs ===

def generate_cactus_graph(n, edge_prob):
    G = nx.erdos_renyi_graph(n, edge_prob)
    G = G.to_undirected()

    G = remove_common_edges(G)

    G = connect_disconnected_nodes(G)

    G = connect_degree_one_nodes(G, 1)

    return G

# Identify edges that appear in more than one cycle
def find_common_edges(G):
    cycles = list(nx.cycle_basis(G))
    edge_count = {}

    # Count the occurrences of each edge in the cycles
    for cycle in cycles:
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            edge = tuple(sorted((u, v)))
            if edge in edge_count:
                edge_count[edge] += 1

            else:
                edge_count[edge] = 1
    return edge_count

# Remove edges that appear in multiple cycles (violating cactus graph definition)
def remove_common_edges(G):
    while True:
        edge_count = find_common_edges(G)

        maxi = 0
        for edge, count in edge_count.items():
            if count > maxi:
                maxi = count
        if maxi > 1:
            G.remove_edge(*edge)
        else:
            break
    return G

# Ensure the graph is connected
def connect_disconnected_nodes(G):
    # Add an edge between nodes in different connected components to ensure the graph is connected
    components = list(nx.connected_components(G))
    while not nx.is_connected(G):
        u = random.choice(list(components[0]))
        v = random.choice(list(components[1]))
        G.add_edge(u, v)
        components = list(nx.connected_components(G))
    return G

# Connect degree-one nodes in a controlled way (only if they don't create a second path)
def connect_degree_one_nodes(G, pr):
    degree_one_nodes = [node for node, degree in G.degree() if degree == 1]

    num_to_keep = int(len(degree_one_nodes) * pr)

    selected_nodes = degree_one_nodes[:num_to_keep]
    random.shuffle(selected_nodes)

    # Créer des arêtes entre les sommets de la liste
    for i in range(0, len(selected_nodes) - 1, 2):
        u = selected_nodes[i]
        v = selected_nodes[i + 1]
        all_paths = list(nx.all_simple_paths(G, source=u, target=v))
        if len(all_paths) == 1:
            G.add_edge(u, v)
    return G

# === Function to generate cycle graphs ===

def generate_cycle_graph(n):
    # Create an empty graph and add n nodes
    G = nx.Graph()
    G.add_nodes_from(range(n))
    nodes = list(G.nodes())
    np.random.shuffle(nodes)

    # Add edges to form a cycle connecting the nodes
    edges = [(nodes[i], nodes[(i + 1) % n]) for i in range(n)]
    G.add_edges_from(edges)

    return G

# === Function to generate binary trees ===

# Generate a tree where each new node connects to a random existing node.
def generate_tree_graph(n_nodes):
    G = nx.Graph()
    G.add_node(0)  

    for i in range(1, n_nodes):
        node_to_connect = random.choice(list(G.nodes))
        G.add_node(i)
        G.add_edge(i, node_to_connect)
    
    return G

# === Function to generate binary trees ===

#Generate a binary tree where each node can have at most 2 children.
def generate_binary_tree(n_nodes):
    G = nx.Graph()
    G.add_node(0)

    # List of candidate nodes that can be connected to (initially just the root)
    available_nodes = [0]
    
    # Dictionary to track the number of times a parent has been chosen
    parent_choices = {0: 0}
    
    for i in range(1, n_nodes):
        parent = random.choice(available_nodes)
        G.add_node(i)
        G.add_edge(i, parent)
        
        if parent not in parent_choices:
            parent_choices[parent] = 0
        
        parent_choices[parent] += 1
        
        if parent_choices[parent] == 2:
            available_nodes.remove(parent)
        
        if parent_choices.get(i, 0) < 2:
            available_nodes.append(i)

    return G

# === Main function to create adjacency matrices ===

def create_adjacency_matrices(output_dir_matrices, graph_type, n_graphs=1000, n_nodes=10, edge_prob=0.5):
    os.makedirs(output_dir_matrices, exist_ok=True)  

    # Mapping graph types to their generator functions
    graph_generators = {
        1: lambda: generate_planar_graph(n_nodes, edge_prob),
        2: lambda: generate_cactus_graph(n_nodes, edge_prob),
        3: lambda: generate_cycle_graph(n_nodes),
        4: lambda: generate_tree_graph(n_nodes),
        5: lambda: generate_binary_tree(n_nodes)
    }

    if graph_type not in graph_generators:
        raise ValueError(f"Type de graphe invalide: {graph_type}. Types disponibles: {list(graph_generators.keys())}")

    for i in range(n_graphs):
        G = graph_generators[graph_type]()
        adj_matrix = nx.to_numpy_array(G, dtype=int)
        matrix_filename = os.path.join(output_dir_matrices, f"graph_{i + 1}.txt")
        np.savetxt(matrix_filename, adj_matrix, fmt='%d')

        print(f"Matrice d'adjacence sauvegardée: {matrix_filename}")

    print(f"{n_graphs} matrices d'adjacence générées et sauvegardées dans '{output_dir_matrices}'.")


# === Argparse command-line interface ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Générer des matrices d'adjacence de graphes.")
    parser.add_argument("--graph_type", type=int, required=True, 
                        help="Type de graphe : 1=planaire, 2=cactus, 3=cycle, 4=arbre; 5=arbre binaire.")
    parser.add_argument("--n_graphs", type=int, default=1000, 
                        help="Nombre de graphes à générer. (Défaut : 1000)")
    parser.add_argument("--n_nodes", type=int, default=10, 
                        help="Nombre de nœuds par graphe. (Défaut : 10)")
    parser.add_argument("--edge_prob", type=float, default=0.5, 
                        help="Probabilité d'arête pour les graphes aléatoires (uniquement pour les graphes planaires). (Défaut : 0.5)")
    parser.add_argument("--output_dir", type=str, default="graphs_matrices", 
                        help="Répertoire de sortie pour les matrices d'adjacence. (Défaut : 'graphs_matrices')")

    args = parser.parse_args()

    create_adjacency_matrices(
        output_dir_matrices=args.output_dir,
        graph_type=args.graph_type,
        n_graphs=args.n_graphs,
        n_nodes=args.n_nodes,
        edge_prob=args.edge_prob
    )
