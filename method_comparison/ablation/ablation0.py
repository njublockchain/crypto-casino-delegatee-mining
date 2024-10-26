import networkx as nx
import torch
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import random

# Step 1: Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Set the seed to a fixed number (e.g., 42)

# Step 2: Read GraphML file and prepare data
G = nx.read_graphml('/home/lab0/gnn-wjx/0xc2a81eb482cb4677136d8812cc6db6e0cb580883.graphml')

node_mapping = {node: i for i, node in enumerate(G.nodes())}
node_features = []

# Extract node features
for node, data in G.nodes(data=True):
    feature_vector = [
        data.get('sum_val_in', 0),
        data.get('sum_val_out', 0),
        data.get('avg_val_in', 0),
        data.get('avg_val_out', 0),
        data.get('count_in', 0),
        data.get('count_out', 0),
        data.get('count', 0),
        data.get('freq', 0),
        data.get('freq_in', 0),
        data.get('freq_out', 0),
        data.get('gini_val', 0),
        data.get('gini_val_in', 0),
        data.get('gini_val_out', 0),
        data.get('in_out_rate', 0),
    ]
    node_features.append(feature_vector)

# Convert feature list to numpy array for clustering
node_features = np.array(node_features)

# Step 3: AHC Clustering
def ahc_clustering_with_optimal_silhouette(embeddings, min_clusters=2, max_clusters=10):
    best_n_clusters = None
    best_silhouette_score = -1
    silhouette_scores = []

    for n_clusters in range(min_clusters, max_clusters + 1):
        ahc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='average')
        cluster_labels = ahc.fit_predict(embeddings)

        silhouette_avg = silhouette_score(embeddings, cluster_labels, metric='euclidean')
        silhouette_scores.append(silhouette_avg)

        print(f"Number of clusters: {n_clusters}, Silhouette score: {silhouette_avg}")

        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_n_clusters = n_clusters

    return best_n_clusters

# Step 4: Apply AHC to node features
best_n_clusters = ahc_clustering_with_optimal_silhouette(node_features)

# Final clustering with the optimal number of clusters
ahc = AgglomerativeClustering(n_clusters=best_n_clusters, affinity='euclidean', linkage='average')
best_clusters = ahc.fit_predict(node_features)

# Step 5: Output the final clustering result
final_clusters = {}
for idx, label in enumerate(best_clusters):
    if label not in final_clusters:
        final_clusters[label] = []
    final_clusters[label].append(idx)

print(f"\nFinal Community Assignments with {best_n_clusters} clusters:")
for community, nodes in final_clusters.items():
    print(f"Community {community}: Nodes {nodes}")

# Step 6: Save the best embeddings and community assignments back to the graph
for i, node in enumerate(G.nodes()):
    # Add community label to the node
    G.nodes[node]['community'] = best_clusters[i]

# Write updated graph with community labels to a file
nx.write_graphml(G, '/home/lab0/gnn-wjx/0xc2a81eb482cb4677136d8812cc6db6e0cb580883_ablation1_all_cluster_best_loss.graphml')