import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from itertools import combinations  # 导入 combinations

# Step 1: Read GraphML file and prepare data for GAT
G = nx.read_graphml('/home/lab0/gnn-wjx/0xc2a81eb482cb4677136d8812cc6db6e0cb580883.graphml')

node_mapping = {node: i for i, node in enumerate(G.nodes())}
node_features = []


# 遍历每个节点，正确提取其特征
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

# 将特征列表转换为 PyTorch tensor
node_features = torch.tensor(node_features, dtype=torch.float)


edge_index = []
edge_weights = []
for u, v, data in G.edges(data=True):
    edge_index.append([node_mapping[u], node_mapping[v]])
    edge_weights.append(data.get('transfer_value', 1.0))
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
edge_weights = torch.tensor(edge_weights, dtype=torch.float)

# Step 2: GAT Model Definition
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=num_heads, dropout=0.5)
        self.conv2 = GATConv(8 * num_heads, out_channels, heads=1, concat=False, dropout=0.5)

    def forward(self, x, edge_index, edge_weights):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.dropout(x, p=0.6, train=self.training)
        x = self.conv2(x, edge_index)
        return x

# Step 3: LSTM Model Definition
class TLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, sequence_length):
        super(TLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, embedding_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        embedding = self.fc(lstm_out[:, -1, :])
        return embedding

# Prepare sequences for each node (example based on your LSTM code)
# def prepare_sequences(graph, S):
#     sequences = {}
#     for node in graph.nodes(data=True):
#         timestamps = []
#         for u, v, data in graph.edges(node[0], data=True):
#             if 'block_number' in data:
#                 if v == node[0]:
#                     timestamps.append(data['block_number'])
#                 elif u == node[0]:
#                     timestamps.append(-data['block_number'])
#         timestamps = sorted(timestamps)
#         m = len(timestamps)
#         if m < S:
#             z_u_prime = np.pad(timestamps, (0, S - m), 'constant', constant_values=0)
#         else:
#             z_u_prime = np.array(timestamps[:S])
#         sequences[node[0]] = z_u_prime.astype(float)
#     return sequences

def prepare_sequences(graph, S):
    sequences = {}
    for node in graph.nodes(data=True):
        # Extract block numbers and transfer values, apply sign based on edge direction
        transactions = []
        for u, v, data in graph.edges(node[0], data=True):
            if 'block_number' in data and 'transfer_value' in data:
                # Incoming edge: target is the current node (v = node)
                if v == node[0]:
                    transactions.append((data['block_number'], data['transfer_value']))
                # Outgoing edge: source is the current node (u = node)
                elif u == node[0]:
                    transactions.append((data['block_number'], -data['transfer_value']))

        # Sort transactions by block number and extract transfer values
        transactions.sort(key=lambda x: x[0])  # Sort by block number
        transfer_values = [t[1] for t in transactions]  # Extract transfer values

        # Prepare z_u_prime
        m = len(transfer_values)
        if m < S:
            z_u_prime = np.pad(transfer_values, (0, S - m), 'constant', constant_values=0)
        else:
            z_u_prime = np.array(transfer_values[:S])

        # Store the sequence
        sequences[node[0]] = z_u_prime.astype(float)  # Store the sequence as float array

    return sequences

# Step 4: Self-Supervised Loss
def ssl_loss(embeddings, clusters):
    loss_intra = 0
    loss_inter = 0
    for C_k in clusters:
        for i in C_k:
            for j in C_k:
                loss_intra += torch.norm(embeddings[i] - embeddings[j]) ** 2
    for C_k, C_l in combinations(clusters, 2):
        for i in C_k:
            for j in C_l:
                loss_inter += torch.norm(embeddings[i] - embeddings[j]) ** 2
    intra_cluster_term = loss_intra / len(clusters)
    inter_cluster_term = loss_inter / (len(clusters) * (len(clusters) - 1))
    return intra_cluster_term - inter_cluster_term

# Step 5: Structural Entropy (SE) Calculation (Simplified example)
def calculate_2d_se(graph, embeddings):
    SE = 0
    vol = sum(nx.get_edge_attributes(graph, 'transfer_value').values())
    for node in graph.nodes():
        d = graph.in_degree(node, weight='transfer_value')
        if d > 0:
            d_tensor = torch.tensor(d, dtype=torch.float32)
            vol_tensor = torch.tensor(vol, dtype=torch.float32)
            SE += -(d_tensor / vol_tensor) * torch.log2(d_tensor / vol_tensor)
    return SE

# Step 6: AHC Clustering
def ahc_clustering_with_optimal_silhouette(embeddings, min_clusters=2, max_clusters=10):
    best_n_clusters = None
    best_silhouette_score = -1
    silhouette_scores = []

    for n_clusters in range(min_clusters, max_clusters + 1):
        ahc = AgglomerativeClustering(n_clusters=n_clusters, affinity='cosine', linkage='average')
        cluster_labels = ahc.fit_predict(embeddings)

        silhouette_avg = silhouette_score(embeddings, cluster_labels, metric='cosine')
        silhouette_scores.append(silhouette_avg)

        print(f"Number of clusters: {n_clusters}, Silhouette score: {silhouette_avg}")

        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_n_clusters = n_clusters

    return best_n_clusters

# Step 7: Combined Loss Function
def combined_loss(gat_loss, lstm_loss, se_loss, ssl_loss, weights):
    return weights['gat'] * gat_loss + weights['lstm'] * lstm_loss + weights['se'] * se_loss + weights['ssl'] * ssl_loss

# Step 8: Model Initialization
gat_model = GAT(in_channels=node_features.shape[1], out_channels=14)  # GAT embedding
lstm_model = TLSTM(input_size=1, hidden_size=64, embedding_size=10, sequence_length=10)  # LSTM embedding
gat_optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.005)
lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.005)

# Prepare sequences for LSTM
sequences = prepare_sequences(G, 10)
sequence_data_np = np.array([sequences[node] for node in G.nodes()])  # 转换为 numpy array
sequence_data = torch.tensor(sequence_data_np, dtype=torch.float32).unsqueeze(2)  # 转换为 tensor 并添加维度

# Step 9: Training Loop
for epoch in range(500):
    gat_optimizer.zero_grad()
    lstm_optimizer.zero_grad()

    # GAT embeddings
    gat_embeddings = gat_model(node_features, edge_index, edge_weights)

    # LSTM embeddings
    lstm_embeddings = lstm_model(sequence_data)

    # Concatenate embeddings from GAT and LSTM
    concatenated_embeddings = torch.cat((gat_embeddings, lstm_embeddings), dim=1)

    # Step 7: 选择最优聚类数
    best_n_clusters = ahc_clustering_with_optimal_silhouette(concatenated_embeddings.detach().cpu().numpy())

    # Step 8: 使用最优聚类数进行最终聚类
    ahc = AgglomerativeClustering(n_clusters=best_n_clusters, affinity='cosine', linkage='average')
    cluster_labels = ahc.fit_predict(concatenated_embeddings.detach().cpu().numpy())

    # 输出聚类结果
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)

    print(f"Epoch {epoch}, Best number of clusters: {best_n_clusters}, Final clusters: {clusters}")

    # Cluster-based SSL Loss
    ssl_loss_value = ssl_loss(concatenated_embeddings, clusters.values())

    # Calculate GAT Loss (MSE)
    gat_loss = torch.nn.functional.mse_loss(gat_embeddings, node_features)

    # Calculate LSTM Loss (Reconstruction loss, placeholder)
    lstm_loss = torch.nn.functional.mse_loss(lstm_embeddings, sequence_data.squeeze(2))

    # Calculate SE Loss
    se_loss = calculate_2d_se(G, concatenated_embeddings)

    # Combine Losses
    final_loss = combined_loss(gat_loss, lstm_loss, se_loss, ssl_loss_value, weights={'gat': 1.0, 'lstm': 1.0, 'se': 0.5, 'ssl': 0.5})

    # Backpropagation
    final_loss.backward()
    gat_optimizer.step()
    lstm_optimizer.step()

    print(f'Epoch {epoch}, Loss: {final_loss.item()}')

# Step 10: Save and Output Final Community Assignments

# 使用最优聚类数再次执行 AHC 聚类
best_n_clusters = ahc_clustering_with_optimal_silhouette(concatenated_embeddings.detach().cpu().numpy())

# 使用最优聚类数进行最终聚类
ahc = AgglomerativeClustering(n_clusters=best_n_clusters, affinity='cosine', linkage='average')
cluster_labels = ahc.fit_predict(concatenated_embeddings.detach().cpu().numpy())

# 输出聚类结果
final_clusters = {}
for idx, label in enumerate(cluster_labels):
    if label not in final_clusters:
        final_clusters[label] = []
    final_clusters[label].append(idx)

print(f"\nFinal Community Assignments with {best_n_clusters} clusters:")
for community, nodes in final_clusters.items():
    print(f"Community {community}: Nodes {nodes}")

# Save the final embeddings and community assignments back to the graph
embeddings = concatenated_embeddings.detach().numpy()
for i, node in enumerate(G.nodes()):
    for j, val in enumerate(embeddings[i]):
        G.nodes[node][f'embedding_{j}'] = val
    # Add community label to the node
    G.nodes[node]['community'] = cluster_labels[i]

# Write updated graph with community labels to a file
nx.write_graphml(G, '/home/lab0/gnn-wjx/0xc2a81eb482cb4677136d8812cc6db6e0cb580883_all_cluster.graphml')