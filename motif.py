import networkx as nx
from itertools import permutations, combinations
import numpy as np

def find_triad_motifs_digraph(DG):
    motifs = {}

    for n1, n2, n3 in combinations(DG.nodes(), 3):
        edges = [(ni, nj) for ni, nj in permutations([n1, n2, n3], 2) if DG.has_edge(ni, nj)]
        if len(edges) >= 2:
            sg = DG.subgraph([n1, n2, n3]).copy()
            matrix = nx.to_numpy_array(sg)
            matrix_tuple = tuple(map(tuple, matrix.tolist()))
            motifs[matrix_tuple] = motifs.get(matrix_tuple, 0) + 1

    return motifs

def calculate_zscore(DG, num_randomizations=1000):
    original_motifs = find_triad_motifs_digraph(DG)
    random_motifs_counts = {motif: [] for motif in original_motifs.keys()}

    for _ in range(num_randomizations):
        random_DG = nx.DiGraph(nx.random_reference(DG.to_undirected(), connectivity=False))
        random_motifs = find_triad_motifs_digraph(random_DG)
        for motif in random_motifs_counts.keys():
            random_motifs_counts[motif].append(random_motifs.get(motif, 0))

    z_scores = {}
    for motif, counts in random_motifs_counts.items():
        mean = np.mean(counts)
        std = np.std(counts)
        z_scores[motif] = (original_motifs[motif] - mean) / std

    return z_scores


filename = ""
G = nx.read_graphml(f'{filename}.graphml')

z_scores = calculate_zscore(G)
for motif, z in z_scores.items():
    print(f"Motif {motif} has z-score {z:.2f}")
