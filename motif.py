import networkx as nx
from itertools import permutations, combinations
import numpy as np

# Function to find triad motifs (3-node subgraphs) in a directed graph
def find_triad_motifs_digraph(DG):
    motifs = {}  # Store detected motifs and their counts

    # Iterate over all possible combinations of 3 nodes
    for n1, n2, n3 in combinations(DG.nodes(), 3):
        # Check for directed edges between the node combinations
        edges = [(ni, nj) for ni, nj in permutations([n1, n2, n3], 2) if DG.has_edge(ni, nj)]
        
        # Only consider if there are 2 or more directed edges in the triad
        if len(edges) >= 2:
            sg = DG.subgraph([n1, n2, n3]).copy()
            matrix = nx.to_numpy_array(sg)  # Convert subgraph to adjacency matrix
            matrix_tuple = tuple(map(tuple, matrix.tolist()))  # Convert matrix to tuple for hashing
            
            # Count occurrences of motifs
            motifs[matrix_tuple] = motifs.get(matrix_tuple, 0) + 1

    return motifs

# Function to calculate the z-scores for triad motifs by comparing to random graphs
def calculate_zscore(DG, num_randomizations=1000):
    original_motifs = find_triad_motifs_digraph(DG)  # Find motifs in the original graph
    random_motifs_counts = {motif: [] for motif in original_motifs.keys()}  # Store counts from random graphs

    # Generate random graphs and count motifs
    for _ in range(num_randomizations):
        random_DG = nx.DiGraph(nx.random_reference(DG.to_undirected(), connectivity=False))
        random_motifs = find_triad_motifs_digraph(random_DG)
        
        for motif in random_motifs_counts.keys():
            random_motifs_counts[motif].append(random_motifs.get(motif, 0))

    # Calculate z-scores for each motif
    z_scores = {}
    for motif, counts in random_motifs_counts.items():
        mean = np.mean(counts)
        std = np.std(counts)
        z_scores[motif] = (original_motifs[motif] - mean) / std

    return z_scores

# Read in a directed graph from a GraphML file
filename = "0xc2a81eb482cb4677136d8812cc6db6e0cb580883_complete"
G = nx.read_graphml(f'{filename}.graphml')

# Calculate z-scores and print results
z_scores = calculate_zscore(G)
for motif, z in z_scores.items():
    print(f"Motif {motif} has z-score {z:.2f}")
