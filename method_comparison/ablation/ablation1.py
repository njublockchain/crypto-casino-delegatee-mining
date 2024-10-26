import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.nn import GATConv
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

# Step 2: Read GraphML file and prepare data for GAT
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

# Convert feature list to PyTorch tensor
node_features = torch.tensor(node_features, dtype=torch.float)

edge_index = []
edge_weights = []
for u, v, data in G.edges(data=True):
    edge_index.append([node_mapping[u], node_mapping[v]])
    edge_weights.append(data.get('transfer_value', 1.0))
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
edge_weights = torch.tensor(edge_weights, dtype=torch.float)

# Step 3: GAT Model Definition
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=num_heads, dropout=0.5)
        self.conv2 = GATConv(8 * num_heads, out_channels, heads=1, concat=False, dropout=0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.dropout(x, p=0.6, train=self.training)
        x = self.conv2(x, edge_index)
        return x

# Step 4: AHC Clustering
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

# Step 5: Model Initialization
gat_model = GAT(in_channels=node_features.shape[1], out_channels=14)  # GAT embedding
gat_optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.005)

# Step 6: Initialize variables to store the best result
best_loss = float('inf')  # Start with a very high value for loss
best_embeddings = None
best_clusters = None
best_n_clusters = None

# Step 7: Training Loop
for epoch in range(200):
    gat_optimizer.zero_grad()

    # GAT embeddings
    gat_embeddings = gat_model(node_features, edge_index)

    # Step 8: Optimal clustering
    current_n_clusters = ahc_clustering_with_optimal_silhouette(gat_embeddings.detach().cpu().numpy())

    # Final clustering
    ahc = AgglomerativeClustering(n_clusters=current_n_clusters, affinity='euclidean', linkage='average')
    cluster_labels = ahc.fit_predict(gat_embeddings.detach().cpu().numpy())

    # Calculate GAT Loss (MSE)
    gat_loss = torch.nn.functional.mse_loss(gat_embeddings, node_features)

    # Check if this is the best loss so far
    if gat_loss.item() < best_loss:
        best_loss = gat_loss.item()
        best_embeddings = gat_embeddings.detach().cpu().numpy()
        best_clusters = cluster_labels
        best_n_clusters = current_n_clusters

    # Backpropagation
    gat_loss.backward()
    gat_optimizer.step()

    print(f'Epoch {epoch}, Loss: {gat_loss.item()}, Best number of clusters: {current_n_clusters}')

# Step 9: Output the best result based on the smallest loss
print(f"\nBest Loss: {best_loss}")
print(f"Best number of clusters: {best_n_clusters}")

# Output final clusters
final_clusters = {}
for idx, label in enumerate(best_clusters):
    if label not in final_clusters:
        final_clusters[label] = []
    final_clusters[label].append(idx)

print(f"\nFinal Community Assignments with {best_n_clusters} clusters:")
for community, nodes in final_clusters.items():
    print(f"Community {community}: Nodes {nodes}")

# Step 10: Save the best embeddings and community assignments back to the graph
for i, node in enumerate(G.nodes()):
    for j, val in enumerate(best_embeddings[i]):
        G.nodes[node][f'embedding_{j}'] = val
    # Add community label to the node
    G.nodes[node]['community'] = best_clusters[i]

# Write updated graph with community labels to a file
nx.write_graphml(G, '/home/lab0/gnn-wjx/0xc2a81eb482cb4677136d8812cc6db6e0cb580883_ablation1_all_cluster_best_loss.graphml')

