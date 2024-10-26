# LSTM的layer
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from itertools import combinations

# Step 1: Read GraphML file and prepare data for GAT
G = nx.read_graphml('/home/lab0/gnn-wjx/0xc2a81eb482cb4677136d8812cc6db6e0cb580883.graphml')

node_mapping = {node: i for i, node in enumerate(G.nodes())}
node_features = []

# 提取节点特征
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

# 提取边信息
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
    def __init__(self, input_size, hidden_size, embedding_size, sequence_length, num_layers=1):
        super(TLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, embedding_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        embedding = self.fc(lstm_out[:, -1, :])
        return embedding

# Step 4: Prepare sequences for LSTM
def prepare_sequences(graph, S):
    sequences = {}
    for node in graph.nodes(data=True):
        transactions = []
        for u, v, data in graph.edges(node[0], data=True):
            if 'block_number' in data and 'transfer_value' in data:
                if v == node[0]:
                    transactions.append((data['block_number'], data['transfer_value']))
                elif u == node[0]:
                    transactions.append((data['block_number'], -data['transfer_value']))

        transactions.sort(key=lambda x: x[0])  # Sort by block number
        transfer_values = [t[1] for t in transactions]

        m = len(transfer_values)
        if m < S:
            z_u_prime = np.pad(transfer_values, (0, S - m), 'constant', constant_values=0)
        else:
            z_u_prime = np.array(transfer_values[:S])

        sequences[node[0]] = z_u_prime.astype(float)

    return sequences

# Step 5: Self-Supervised Loss
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

# Step 6: Structural Entropy (SE) Calculation
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

# Step 7: AHC Clustering
def ahc_clustering_with_optimal_silhouette(embeddings, min_clusters=2, max_clusters=10):
    best_n_clusters = None
    best_silhouette_score = -1

    for n_clusters in range(min_clusters, max_clusters + 1):
        ahc = AgglomerativeClustering(n_clusters=n_clusters, affinity='cosine', linkage='average')
        cluster_labels = ahc.fit_predict(embeddings)

        silhouette_avg = silhouette_score(embeddings, cluster_labels, metric='cosine')

        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_n_clusters = n_clusters

    return best_n_clusters

# Step 8: Combined Loss Function
def combined_loss(gat_loss, lstm_loss, se_loss, ssl_loss, weights):
    return weights['gat'] * gat_loss + weights['lstm'] * lstm_loss + weights['se'] * se_loss + weights['ssl'] * ssl_loss

# 实验：设置不同的 LSTM 层数并输出结果
lstm_layers = [1, 2, 3, 4]  # 要测试的 LSTM 层数

# 用于存储所有层数的聚类结果
all_results = {}

for num_layers in lstm_layers:
    print(f"\nTesting with {num_layers} LSTM layers")
    
    # Step 9: Initialize GAT and LSTM models with the given number of LSTM layers
    gat_model = GAT(in_channels=node_features.shape[1], out_channels=14)  # GAT embedding
    lstm_model = TLSTM(input_size=1, hidden_size=64, embedding_size=10, sequence_length=10, num_layers=num_layers)  # LSTM embedding
    gat_optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.005)
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.005)
    
    # Prepare sequences for LSTM
    sequences = prepare_sequences(G, 10)
    sequence_data_np = np.array([sequences[node] for node in G.nodes()])
    sequence_data = torch.tensor(sequence_data_np, dtype=torch.float32).unsqueeze(2)  # 添加维度
    
    # Step 10: Training Loop
    for epoch in range(100):  # 设置为 100 个 epoch
        gat_optimizer.zero_grad()
        lstm_optimizer.zero_grad()

        # GAT embeddings
        gat_embeddings = gat_model(node_features, edge_index, edge_weights)

        # LSTM embeddings
        lstm_embeddings = lstm_model(sequence_data)

        # Concatenate embeddings from GAT and LSTM
        concatenated_embeddings = torch.cat((gat_embeddings, lstm_embeddings), dim=1)

        # AHC clustering and evaluation
        best_n_clusters = ahc_clustering_with_optimal_silhouette(concatenated_embeddings.detach().cpu().numpy())

        # Cluster-based SSL Loss
        ahc = AgglomerativeClustering(n_clusters=best_n_clusters, affinity='cosine', linkage='average')
        cluster_labels = ahc.fit_predict(concatenated_embeddings.detach().cpu().numpy())

        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        # Calculate losses
        ssl_loss_value = ssl_loss(concatenated_embeddings, clusters.values())
        gat_loss = torch.nn.functional.mse_loss(gat_embeddings, node_features)
        lstm_loss = torch.nn.functional.mse_loss(lstm_embeddings, sequence_data.squeeze(2))
        se_loss = calculate_2d_se(G, concatenated_embeddings)

        final_loss = combined_loss(gat_loss, lstm_loss, se_loss, ssl_loss_value, 
                                   weights={'gat': 1.0, 'lstm': 1.0, 'se': 0.5, 'ssl': 0.5})

        # Backpropagation and optimization steps
        final_loss.backward()
        gat_optimizer.step()
        lstm_optimizer.step()

        # Print progress for the current epoch
        print(f'Epoch {epoch}, Loss: {final_loss.item()}')

    # 保存每个 LSTM 层数的聚类结果到 all_results 中
    all_results[num_layers] = clusters

# 最后统一输出所有层数的聚类结果
for num_layers, clusters in all_results.items():
    print(f"\nFinal clustering results for {num_layers} LSTM layers:")
    for community, nodes in clusters.items():
        print(f"Community {community}: Nodes {nodes}")


# Final clustering results for 1 LSTM layers:
# Community 2: Nodes [0, 1, 2, 3, 7, 10, 16, 43, 44, 66, 67, 90, 96, 161, 174, 180, 191, 210, 229, 235, 239, 249, 253]
# Community 6: Nodes [4, 236]
# Community 3: Nodes [5, 6, 11, 12, 13, 15, 17, 18, 20, 21, 24, 28, 29, 30, 32, 34, 37, 39, 40, 41, 46, 47, 48, 49, 50, 51, 53, 54, 55, 58, 59, 60, 61, 62, 63, 65, 68, 69, 70, 74, 76, 77, 78, 80, 81, 82, 84, 87, 93, 94, 97, 99, 100, 101, 102, 103, 104, 105, 106, 109, 110, 111, 113, 118, 119, 121, 125, 129, 130, 133, 134, 135, 136, 137, 138, 140, 141, 143, 144, 146, 147, 148, 150, 151, 152, 155, 156, 160, 162, 164, 165, 166, 167, 169, 171, 175, 176, 179, 181, 185, 186, 187, 188, 190, 192, 193, 194, 195, 196, 197, 200, 202, 204, 205, 206, 208, 209, 214, 216, 217, 218, 220, 221, 223, 224, 225, 226, 227, 228, 230, 232, 233, 234, 237, 241, 242, 243, 244, 248, 250, 251, 254, 259, 260, 261, 262, 263, 265, 267, 268, 269, 271]
# Community 0: Nodes [8, 14, 23, 36, 52, 64, 71, 73, 88, 89, 92, 107, 112, 123, 131, 142, 159, 168, 172, 189, 211, 212, 222, 231, 240, 252, 255, 264, 270, 273]
# Community 1: Nodes [9, 31, 95, 120, 122, 153, 170, 173, 201, 207, 238, 247, 256, 257]
# Community 8: Nodes [19, 22, 25, 26, 27, 33, 35, 38, 42, 83, 86, 108, 114, 115, 116, 117, 154, 157, 184, 199, 213, 219, 245, 246]
# Community 4: Nodes [45, 56, 57, 85, 98, 126, 127, 132, 149, 158, 163, 177, 182, 183, 198, 203, 272]
# Community 7: Nodes [72, 75, 79, 128, 139, 145, 178, 258, 266]
# Community 9: Nodes [91, 215]
# Community 5: Nodes [124]

# Final clustering results for 2 LSTM layers:
# Community 8: Nodes [0, 1, 2, 4, 5, 6, 7, 9, 11, 14, 15, 17, 20, 23, 24, 25, 27, 28, 30, 32, 33, 34, 38, 40, 41, 43, 44, 46, 47, 49, 52, 53, 54, 56, 58, 59, 60, 62, 64, 65, 66, 69, 71, 72, 76, 78, 82, 83, 84, 85, 86, 87, 89, 94, 95, 96, 97, 98, 100, 103, 105, 106, 108, 109, 110, 112, 113, 114, 116, 117, 120, 122, 123, 127, 130, 132, 133, 137, 140, 143, 144, 145, 147, 148, 149, 150, 155, 158, 159, 160, 161, 162, 163, 164, 165, 166, 169, 170, 171, 173, 174, 175, 176, 177, 178, 180, 183, 184, 186, 187, 188, 189, 192, 193, 195, 196, 197, 204, 206, 207, 208, 209, 211, 212, 213, 214, 216, 218, 219, 220, 223, 224, 225, 226, 228, 229, 230, 232, 233, 234, 235, 236, 237, 241, 245, 246, 248, 251, 255, 258, 260, 262, 264, 265, 267, 268, 270, 271, 272]
# Community 1: Nodes [3, 8, 16, 18, 22, 26, 31, 35, 36, 50, 51, 67, 79, 80, 93, 99, 102, 104, 115, 135, 136, 142, 154, 167, 200, 201, 202, 217, 222, 227, 231, 238, 240, 247, 252, 257, 263, 266, 269, 273]
# Community 3: Nodes [10, 19, 21, 39, 55, 61, 75, 90, 107, 118, 121, 124, 125, 134, 168, 172, 179, 205, 221, 250, 254, 259]
# Community 2: Nodes [12, 91, 111, 138, 185, 198, 210, 215]
# Community 0: Nodes [13, 29, 37, 42, 45, 57, 63, 68, 70, 74, 81, 128, 129, 131, 139, 141, 151, 152, 153, 156, 182, 191, 203, 242, 243, 244, 249]
# Community 9: Nodes [48, 88, 126, 181]
# Community 5: Nodes [73, 194, 256]
# Community 4: Nodes [77, 101, 199, 261]
# Community 7: Nodes [92, 146]
# Community 6: Nodes [119, 157, 190, 239, 253]

# Final clustering results for 3 LSTM layers:
# Community 1: Nodes [0, 6, 20, 99, 161]
# Community 7: Nodes [1, 9, 45, 50, 58, 61, 73, 75, 80, 92, 183, 187, 191, 259]
# Community 5: Nodes [2, 3, 4, 5, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18, 19, 23, 24, 25, 29, 30, 31, 32, 33, 38, 40, 41, 43, 44, 46, 47, 49, 51, 52, 53, 55, 56, 59, 62, 63, 65, 66, 71, 72, 76, 77, 79, 81, 82, 85, 86, 87, 88, 90, 91, 93, 94, 96, 97, 98, 100, 102, 107, 108, 109, 110, 112, 114, 119, 120, 121, 122, 123, 124, 125, 126, 129, 130, 131, 132, 133, 135, 137, 138, 139, 141, 142, 144, 146, 148, 149, 152, 153, 155, 156, 158, 159, 162, 164, 167, 168, 169, 171, 172, 174, 178, 179, 181, 182, 186, 188, 189, 193, 194, 195, 196, 197, 198, 199, 201, 202, 204, 205, 206, 209, 210, 211, 212, 213, 216, 218, 219, 221, 222, 223, 224, 229, 230, 231, 233, 234, 235, 238, 239, 241, 243, 244, 245, 246, 247, 248, 250, 251, 253, 255, 256, 257, 262, 263, 264, 265, 267, 273]
# Community 8: Nodes [13, 157, 207]
# Community 3: Nodes [21, 34, 39, 64, 101, 127, 150, 165, 226, 242]
# Community 4: Nodes [22, 26, 27, 28, 36, 42, 70, 74, 83, 84, 111, 115, 116, 117, 118, 134, 136, 175, 176, 177, 184, 217, 225, 232, 240, 249, 252, 258, 261, 266, 271]
# Community 6: Nodes [35, 48, 54, 57, 69, 89, 95, 103, 104, 105, 106, 113, 140, 145, 147, 151, 163, 170, 173, 180, 185, 192, 220, 236, 237, 254, 268]
# Community 0: Nodes [37, 68, 78, 128, 166, 190, 203, 208, 227, 228, 269, 270, 272]
# Community 2: Nodes [60, 143, 154, 200, 214, 215, 260]
# Community 9: Nodes [67, 160]

# Final clustering results for 4 LSTM layers:
# Community 3: Nodes [0, 1, 2, 15, 41, 60, 101, 102, 103, 104, 121, 124, 125, 149, 161, 212, 218, 248]
# Community 5: Nodes [3, 4, 7, 8, 10, 12, 13, 17, 18, 19, 20, 21, 23, 31, 32, 33, 44, 47, 48, 49, 53, 54, 55, 58, 63, 64, 65, 66, 67, 68, 70, 73, 80, 81, 86, 87, 88, 89, 90, 91, 92, 93, 94, 98, 99, 100, 105, 107, 108, 115, 116, 117, 119, 122, 123, 128, 129, 130, 131, 133, 137, 138, 139, 140, 141, 142, 143, 146, 147, 148, 150, 153, 154, 155, 157, 158, 159, 160, 162, 163, 164, 166, 167, 168, 170, 171, 173, 175, 178, 181, 183, 185, 195, 196, 197, 201, 204, 206, 207, 208, 209, 210, 211, 216, 217, 219, 221, 223, 225, 226, 228, 229, 230, 232, 237, 239, 243, 244, 247, 257, 258, 259, 261, 262, 264, 265, 266, 268, 270, 273]
# Community 6: Nodes [5, 9, 11, 16, 22, 28, 34, 38, 39, 43, 56, 61, 62, 75, 76, 83, 95, 97, 112, 126, 127, 132, 145, 165, 180, 184, 188, 189, 190, 192, 193, 194, 200, 205, 215, 220, 222, 234, 236, 251]
# Community 9: Nodes [6, 135, 182, 240]
# Community 2: Nodes [14, 96, 109, 134, 144, 187, 253, 254, 260, 272]
# Community 7: Nodes [24, 25, 26, 27, 29, 30, 36, 37, 42, 46, 51, 69, 74, 78, 79, 84, 85, 106, 111, 114, 118, 169, 172, 174, 176, 177, 179, 198, 203, 214, 224, 227, 231, 238, 241, 245, 250, 252, 255, 256, 263, 269]
# Community 0: Nodes [35, 40, 136, 151, 191, 246, 249]
# Community 1: Nodes [45, 57, 59, 82, 113, 156, 186, 213, 235, 242]
# Community 4: Nodes [50, 52, 71, 72, 77, 110, 120, 152, 199, 233, 267, 271]
# Community 8: Nodes [202]
