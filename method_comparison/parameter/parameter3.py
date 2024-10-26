# LSTM的sequence
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

# 修改 GAT 模型定义，使其支持多层
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_heads=4):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_channels, hidden_channels, heads=num_heads, dropout=0.5))
        
        # 中间的GAT层
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=0.5))
        
        # 最后一层
        self.layers.append(GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=0.5))

    def forward(self, x, edge_index, edge_weights):
        for conv in self.layers:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = torch.dropout(x, p=0.6, train=self.training)
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

# 实验：设置不同的 TLSTM 序列长度并输出结果
sequence_lengths = [5, 10, 15, 20, 25]  # 要测试的 TLSTM 序列长度

# 用于存储所有序列长度的聚类结果
all_results = {}

for sequence_length in sequence_lengths:
    print(f"\nTesting with sequence length {sequence_length}")
    
    # Step 9: Initialize GAT and LSTM models
    gat_model = GAT(in_channels=node_features.shape[1], hidden_channels=8, out_channels=14, num_layers=2)  # 固定 GAT 为 2 层
    lstm_model = TLSTM(input_size=1, hidden_size=64, embedding_size=sequence_length, sequence_length=sequence_length, num_layers=2)  # LSTM embedding
    gat_optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.005)
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.005)
    
    # Prepare sequences for LSTM
    sequences = prepare_sequences(G, sequence_length)
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

    # 保存每个序列长度的聚类结果到 all_results 中
    all_results[sequence_length] = clusters

# 最后统一输出所有序列长度的聚类结果
for sequence_length, clusters in all_results.items():
    print(f"\nFinal clustering results for sequence length {sequence_length}:")
    for community, nodes in clusters.items():
        print(f"Community {community}: Nodes {nodes}")

# Final clustering results for sequence length 5:
# Community 7: Nodes [0, 1, 3, 4, 5, 15, 17, 19, 21, 22, 24, 26, 27, 28, 30, 31, 32, 33, 34, 37, 38, 45, 49, 51, 53, 55, 56, 59, 60, 64, 65, 66, 67, 69, 71, 75, 76, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 95, 96, 97, 100, 107, 108, 110, 111, 115, 116, 117, 119, 120, 121, 123, 125, 126, 127, 130, 132, 134, 138, 140, 143, 145, 150, 151, 152, 157, 160, 162, 164, 168, 169, 170, 171, 172, 173, 175, 176, 177, 179, 182, 183, 187, 189, 190, 192, 193, 194, 195, 204, 206, 210, 213, 214, 219, 220, 223, 229, 230, 234, 236, 237, 241, 245, 246, 250, 252, 253, 255, 256, 261, 263, 264, 265, 267, 272, 273]
# Community 2: Nodes [2, 6, 7, 9, 20, 29, 35, 40, 42, 43, 46, 50, 52, 58, 62, 79, 98, 101, 103, 114, 118, 124, 136, 142, 159, 161, 178, 181, 186, 188, 198, 200, 202, 203, 208, 209, 212, 218, 224, 232, 235]
# Community 3: Nodes [8, 14, 70, 77, 102, 104, 155, 211, 215, 217, 231, 249]
# Community 6: Nodes [10, 13, 48, 68, 73, 94, 105, 113, 147, 153, 154, 156, 158, 163, 166, 196, 197, 205, 225, 233, 239, 242, 243, 244, 247, 257, 266, 269, 270]
# Community 8: Nodes [11, 16, 18, 39, 74, 99, 199, 221]
# Community 0: Nodes [12, 23, 25, 44, 57, 63, 72, 78, 87, 106, 109, 112, 122, 128, 129, 137, 139, 149, 165, 174, 180, 185, 191, 201, 216, 222, 226, 227, 228, 238, 248, 251, 254, 259, 260, 262, 268]
# Community 1: Nodes [36, 41, 47, 88, 90, 167, 184, 240, 271]
# Community 9: Nodes [54, 144, 146]
# Community 4: Nodes [61, 93, 131, 133, 135, 141, 148, 207]
# Community 5: Nodes [258]

# Final clustering results for sequence length 10:
# Community 7: Nodes [0, 4, 6, 11, 12, 13, 14, 16, 18, 20, 21, 26, 27, 29, 30, 32, 33, 40, 46, 53, 54, 55, 57, 59, 60, 61, 64, 66, 68, 70, 78, 89, 90, 96, 98, 104, 105, 106, 109, 110, 112, 113, 114, 117, 119, 121, 122, 123, 126, 127, 129, 130, 131, 132, 136, 137, 140, 141, 143, 145, 146, 147, 149, 155, 157, 162, 163, 164, 168, 169, 170, 172, 174, 181, 183, 185, 191, 193, 195, 196, 198, 201, 202, 203, 204, 206, 207, 209, 211, 212, 215, 219, 221, 223, 224, 226, 227, 228, 229, 230, 231, 232, 234, 236, 237, 238, 243, 244, 246, 247, 250, 252, 253, 256, 259, 264, 266, 269, 270, 272, 273]
# Community 0: Nodes [1, 8, 10, 15, 28, 31, 34, 41, 42, 45, 47, 48, 52, 65, 67, 69, 73, 74, 77, 91, 97, 99, 100, 102, 116, 133, 135, 138, 151, 165, 166, 167, 171, 175, 177, 178, 179, 182, 184, 188, 192, 194, 205, 208, 210, 222, 225, 233, 235, 249, 265, 268]
# Community 3: Nodes [2, 5, 19, 22, 24, 36, 37, 49, 50, 56, 62, 63, 80, 81, 87, 107, 111, 115, 118, 125, 128, 134, 142, 153, 158, 159, 160, 173, 180, 197, 199, 213, 216, 217, 218, 220, 242, 245, 248, 254, 260, 263, 271]
# Community 2: Nodes [3, 9, 17, 39, 84, 85, 88, 95, 103, 124, 148, 214, 257, 258, 261]
# Community 6: Nodes [7, 23, 25, 51, 71, 86, 92, 139, 154, 189, 190, 241]
# Community 4: Nodes [35, 43, 44, 58, 72, 82, 93, 94, 101, 108, 150, 156, 161, 176, 186, 200, 239, 251, 255, 262, 267]
# Community 8: Nodes [38, 75, 83]
# Community 9: Nodes [76, 187, 240]
# Community 1: Nodes [79, 120, 152]
# Community 5: Nodes [144]

# Final clustering results for sequence length 15:
# Community 1: Nodes [0, 1, 2, 3, 5, 6, 7, 8, 9, 14, 15, 17, 19, 20, 21, 23, 24, 25, 26, 27, 30, 31, 32, 34, 35, 41, 44, 46, 51, 54, 56, 58, 59, 62, 64, 70, 73, 75, 80, 81, 84, 85, 89, 92, 93, 98, 99, 100, 101, 106, 107, 108, 112, 113, 115, 118, 119, 123, 124, 127, 128, 129, 131, 133, 134, 135, 136, 141, 147, 148, 149, 152, 154, 156, 157, 159, 161, 163, 164, 166, 169, 170, 171, 172, 173, 177, 178, 179, 181, 182, 183, 188, 191, 192, 194, 195, 198, 200, 202, 204, 205, 206, 207, 208, 209, 210, 211, 214, 215, 218, 219, 220, 223, 227, 228, 229, 230, 231, 232, 234, 237, 238, 241, 242, 244, 245, 247, 249, 250, 252, 253, 254, 255, 256, 259, 264, 267, 268, 269, 270, 271, 272, 273]
# Community 0: Nodes [4, 10, 11, 12, 13, 16, 18, 22, 28, 29, 33, 36, 37, 38, 39, 40, 42, 43, 45, 47, 48, 49, 50, 52, 53, 55, 57, 60, 61, 63, 65, 66, 67, 68, 69, 71, 72, 74, 76, 77, 78, 79, 82, 83, 86, 87, 88, 90, 91, 94, 95, 96, 97, 102, 103, 104, 105, 109, 110, 111, 114, 116, 117, 120, 121, 122, 125, 126, 130, 132, 137, 138, 139, 140, 142, 143, 144, 145, 146, 150, 151, 153, 155, 158, 160, 162, 165, 167, 168, 174, 175, 176, 180, 184, 185, 186, 187, 189, 190, 193, 196, 197, 199, 201, 203, 212, 213, 216, 217, 221, 222, 224, 225, 226, 233, 235, 236, 239, 240, 243, 246, 248, 251, 257, 258, 260, 261, 262, 263, 265, 266]

# Final clustering results for sequence length 20:
# Community 7: Nodes [0, 5, 6, 7, 8, 10, 11, 12, 13, 17, 18, 19, 20, 21, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 38, 39, 40, 41, 44, 46, 47, 48, 51, 52, 54, 55, 57, 59, 63, 66, 68, 69, 70, 72, 73, 76, 77, 78, 79, 80, 82, 83, 87, 88, 89, 93, 94, 95, 96, 97, 98, 102, 104, 105, 108, 110, 111, 116, 117, 119, 120, 121, 122, 123, 124, 125, 126, 127, 129, 131, 132, 134, 135, 137, 138, 139, 140, 141, 143, 145, 151, 152, 153, 156, 158, 159, 161, 162, 164, 166, 167, 168, 172, 174, 175, 176, 177, 178, 182, 184, 185, 186, 187, 190, 192, 193, 194, 195, 197, 200, 201, 202, 206, 208, 211, 212, 213, 214, 220, 222, 223, 227, 228, 229, 230, 232, 234, 235, 236, 237, 239, 240, 242, 243, 244, 245, 246, 247, 251, 252, 253, 254, 255, 256, 257, 258, 260, 261, 262, 263, 265, 266, 267, 268, 269, 270, 271]
# Community 0: Nodes [1, 14, 53, 58, 62, 81, 86, 92, 109, 146, 149, 180, 215, 221, 231]
# Community 3: Nodes [2, 27, 50, 100, 112, 148, 163, 210, 259]
# Community 2: Nodes [3, 36, 43, 60, 75, 199]
# Community 4: Nodes [4, 15, 24, 45, 103, 114, 115, 150, 154, 155, 165, 171, 173, 188, 191, 196, 207, 225, 226, 238]
# Community 5: Nodes [9, 16, 30, 64, 67, 84, 128, 142, 233, 250, 272, 273]
# Community 1: Nodes [33, 49, 56, 61, 91, 107, 113, 130, 160, 198, 209, 216, 217, 219, 241]
# Community 8: Nodes [37, 42, 65, 74, 90, 99, 101, 106, 133, 147, 181, 189, 203, 204, 248]
# Community 6: Nodes [71, 118, 169, 183, 218, 249, 264]
# Community 9: Nodes [85, 136, 144, 157, 170, 179, 205, 224]

# Final clustering results for sequence length 25:
# Community 2: Nodes [0, 4, 7, 10, 13, 48, 63, 76, 110, 119, 124, 126, 137, 145, 167, 168, 170, 171, 177, 189, 191, 201, 217, 218, 221, 225, 240, 242, 247, 248, 259, 266, 269]
# Community 7: Nodes [1, 2, 5, 9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 29, 30, 36, 37, 38, 40, 42, 43, 44, 45, 46, 50, 51, 52, 54, 55, 56, 57, 58, 59, 64, 65, 67, 70, 71, 73, 74, 78, 81, 82, 83, 84, 88, 89, 90, 91, 92, 98, 99, 100, 103, 105, 106, 111, 112, 113, 114, 115, 116, 117, 120, 127, 128, 129, 130, 132, 133, 135, 140, 142, 143, 144, 147, 149, 150, 151, 152, 155, 156, 158, 160, 162, 163, 165, 166, 169, 172, 173, 176, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 190, 192, 193, 194, 196, 197, 198, 200, 203, 204, 206, 208, 209, 210, 211, 212, 213, 215, 219, 220, 223, 224, 226, 227, 228, 229, 230, 231, 233, 234, 237, 238, 239, 241, 243, 245, 246, 249, 250, 251, 252, 253, 255, 256, 258, 263, 264, 265, 267, 268, 272, 273]
# Community 5: Nodes [3, 28, 47, 85, 93, 95, 118, 131, 139, 235, 262]
# Community 0: Nodes [6, 8, 16, 21, 32, 34, 53, 61, 69, 77, 80, 96, 102, 109, 125, 141, 161, 199, 202, 205, 214, 216, 222, 236, 257, 270, 271]
# Community 6: Nodes [11, 19, 39, 41, 66, 79, 101, 134, 148, 154, 175, 207, 254]
# Community 9: Nodes [12, 17, 35, 75, 86, 87, 122, 136, 232]
# Community 3: Nodes [31, 104, 108, 121, 157, 174, 260]
# Community 4: Nodes [33, 62, 94, 97, 107, 138]
# Community 1: Nodes [49, 60, 68, 72, 123, 159, 244, 261]
# Community 8: Nodes [146, 153, 164, 195]