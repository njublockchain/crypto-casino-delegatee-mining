# GAT的layer
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

# 实验：设置不同的 GAT 层数并输出结果
gat_layers = [1, 2, 3, 4]  # 要测试的 GAT 层数

# 用于存储所有层数的聚类结果
all_results = {}

for num_layers in gat_layers:
    print(f"\nTesting with {num_layers} GAT layers")
    
    # Step 9: Initialize GAT and LSTM models with the given number of GAT layers
    gat_model = GAT(in_channels=node_features.shape[1], hidden_channels=8, out_channels=14, num_layers=num_layers)  # GAT embedding
    lstm_model = TLSTM(input_size=1, hidden_size=64, embedding_size=10, sequence_length=10, num_layers=2)  # 固定 LSTM 为 2 层
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

    # 保存每个 GAT 层数的聚类结果到 all_results 中
    all_results[num_layers] = clusters

# 最后统一输出所有层数的聚类结果
for num_layers, clusters in all_results.items():
    print(f"\nFinal clustering results for {num_layers} GAT layers:")
    for community, nodes in clusters.items():
        print(f"Community {community}: Nodes {nodes}")

# Final clustering results for 1 GAT layers:
# Community 0: Nodes [0, 1, 3, 6, 17, 19, 28, 30, 34, 35, 62, 71, 73, 99, 109, 111, 113, 118, 120, 126, 141, 146, 160, 167, 179, 182, 190, 207, 210, 217, 219, 224, 226, 227, 235, 239, 261, 264, 266, 267]
# Community 4: Nodes [2, 43, 63, 64, 102, 166, 170, 172, 175, 257, 259, 272]
# Community 7: Nodes [4, 5, 7, 8, 9, 12, 13, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 31, 32, 33, 38, 39, 41, 42, 44, 46, 47, 49, 50, 54, 55, 56, 57, 58, 59, 61, 65, 66, 69, 70, 74, 76, 78, 80, 81, 84, 86, 89, 91, 92, 95, 96, 101, 103, 104, 106, 107, 108, 110, 112, 114, 115, 117, 119, 121, 122, 123, 124, 127, 128, 129, 131, 132, 133, 134, 135, 136, 137, 138, 140, 142, 143, 145, 147, 148, 149, 152, 153, 154, 155, 157, 158, 161, 162, 165, 168, 169, 173, 174, 176, 177, 178, 180, 181, 183, 184, 186, 193, 195, 196, 197, 198, 203, 205, 208, 209, 211, 214, 215, 216, 218, 221, 223, 225, 230, 232, 233, 236, 237, 238, 240, 241, 242, 243, 244, 245, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 258, 260, 265, 269, 271, 273]
# Community 8: Nodes [10, 45, 88, 90, 125, 150, 156, 171, 194, 229, 234]
# Community 2: Nodes [11, 48, 51, 77, 83, 87, 100, 151, 159, 163, 188, 192, 202, 220, 228, 246, 262, 270]
# Community 6: Nodes [23, 27, 29, 36, 53, 60, 130, 185, 191, 199, 201, 263]
# Community 1: Nodes [37, 40, 52, 72, 75, 82, 93, 94, 97, 98, 116, 139, 144, 164, 187, 212, 213, 222]
# Community 5: Nodes [67, 85, 200]
# Community 9: Nodes [68, 79, 105, 189, 268]
# Community 3: Nodes [204, 206, 231]

# Final clustering results for 2 GAT layers:
# Community 4: Nodes [0, 1, 7, 10, 15, 16, 18, 19, 20, 22, 24, 25, 28, 29, 31, 33, 36, 39, 40, 41, 44, 47, 49, 50, 51, 52, 53, 56, 57, 58, 59, 60, 62, 67, 68, 69, 70, 71, 72, 73, 80, 81, 82, 88, 89, 92, 95, 96, 100, 101, 103, 105, 107, 110, 112, 113, 116, 118, 119, 120, 122, 127, 131, 133, 135, 136, 137, 138, 140, 144, 146, 148, 150, 151, 152, 154, 155, 157, 159, 160, 162, 170, 171, 173, 175, 178, 181, 184, 186, 187, 189, 192, 193, 195, 197, 201, 205, 207, 208, 220, 221, 223, 226, 230, 231, 233, 236, 240, 243, 244, 246, 247, 249, 250, 251, 253, 254, 257, 259, 261, 262, 264, 266, 268, 269, 270]
# Community 0: Nodes [2, 3, 11, 17, 21, 43, 45, 46, 48, 54, 55, 61, 63, 64, 75, 83, 87, 90, 94, 97, 102, 104, 111, 121, 123, 124, 126, 130, 143, 145, 149, 156, 158, 161, 164, 167, 168, 176, 182, 183, 188, 191, 194, 200, 206, 216, 217, 218, 219, 222, 225, 229, 235, 237, 242, 248, 256, 260, 272, 273]
# Community 1: Nodes [4, 5, 6, 8, 12, 26, 27, 30, 37, 65, 74, 79, 84, 93, 98, 106, 114, 115, 117, 125, 128, 141, 142, 147, 165, 179, 196, 198, 202, 203, 210, 214, 215, 224, 227, 228, 238, 263, 265]
# Community 6: Nodes [9, 14, 32, 34, 99, 109, 163, 166, 271]
# Community 8: Nodes [13, 35, 91, 153, 169, 267]
# Community 2: Nodes [23, 78, 85, 86, 108, 129, 139, 172, 174, 177, 212, 213, 232, 234, 239, 241, 245, 252]
# Community 3: Nodes [38, 42, 76, 134, 185, 190, 199, 204, 258]
# Community 9: Nodes [66, 132]
# Community 5: Nodes [77, 180, 209]
# Community 7: Nodes [211, 255]

# Final clustering results for 3 GAT layers:
# Community 1: Nodes [0, 8, 17, 39, 107, 110, 111, 127, 147, 149, 167, 214, 257, 264]
# Community 7: Nodes [1, 2, 3, 7, 9, 14, 15, 18, 20, 21, 22, 23, 24, 26, 27, 30, 31, 32, 33, 34, 36, 37, 38, 40, 44, 47, 48, 50, 52, 54, 55, 58, 59, 60, 61, 63, 64, 66, 68, 69, 70, 71, 74, 75, 76, 80, 81, 84, 88, 89, 91, 93, 95, 96, 97, 100, 101, 103, 104, 106, 108, 109, 113, 119, 121, 122, 123, 124, 126, 128, 129, 130, 132, 133, 135, 136, 138, 141, 143, 144, 145, 146, 148, 151, 152, 154, 155, 156, 157, 158, 160, 161, 162, 164, 168, 171, 173, 175, 176, 178, 179, 183, 185, 186, 187, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 204, 205, 206, 207, 209, 210, 211, 215, 219, 221, 222, 223, 224, 225, 226, 230, 231, 232, 233, 234, 235, 236, 237, 238, 240, 242, 247, 250, 251, 252, 256, 259, 263, 265, 266, 268, 271, 273]
# Community 3: Nodes [4, 16, 116, 120, 170, 202, 216, 246, 258]
# Community 0: Nodes [5, 6, 10, 12, 29, 41, 43, 46, 51, 62, 67, 72, 73, 77, 86, 87, 98, 105, 112, 117, 118, 125, 134, 137, 142, 165, 166, 177, 180, 188, 208, 218, 228, 229, 241, 243, 244, 249, 253, 260, 270]
# Community 4: Nodes [11, 53, 83, 85, 90, 92, 131, 159, 163, 213, 217, 220, 267]
# Community 6: Nodes [13, 19, 25, 35, 56, 102, 169, 181, 203, 239, 262]
# Community 9: Nodes [28, 78, 79, 114, 115, 172, 245]
# Community 5: Nodes [42, 49, 57, 65, 82, 94, 99, 139, 140, 150, 153, 174, 200, 201, 212, 254, 261, 269]
# Community 8: Nodes [45, 182, 227]
# Community 2: Nodes [184, 248, 255, 272]

# Final clustering results for 4 GAT layers:
# Community 4: Nodes [0, 24, 33, 56, 72, 103, 162, 175, 185, 194, 198, 242, 252]
# Community 7: Nodes [1, 4, 10, 11, 12, 13, 15, 16, 18, 27, 28, 31, 35, 38, 39, 40, 41, 46, 52, 54, 58, 60, 62, 63, 64, 65, 67, 68, 70, 71, 73, 75, 76, 80, 82, 83, 87, 89, 90, 91, 92, 94, 97, 98, 100, 101, 105, 108, 110, 111, 114, 115, 116, 118, 121, 122, 125, 126, 127, 129, 131, 133, 134, 136, 137, 138, 140, 141, 142, 144, 145, 146, 147, 149, 151, 152, 154, 155, 158, 160, 161, 165, 166, 167, 168, 170, 171, 174, 178, 179, 180, 181, 182, 186, 187, 190, 191, 193, 195, 196, 197, 199, 200, 201, 204, 206, 210, 211, 213, 214, 215, 216, 217, 218, 219, 221, 222, 223, 226, 228, 229, 230, 231, 233, 234, 235, 236, 239, 244, 245, 246, 248, 250, 251, 253, 254, 256, 257, 258, 260, 261, 263, 264, 269, 271, 272, 273]
# Community 9: Nodes [2, 17, 21, 25, 26, 49, 61, 69, 74, 78, 84, 106, 117, 163, 173, 177, 209, 225, 227, 265]
# Community 2: Nodes [3, 9, 14, 22, 32, 34, 50, 57, 99, 102, 104, 107, 124, 135, 139, 148, 150, 153, 205, 207, 232, 237, 240, 262]
# Community 8: Nodes [5, 8, 88, 189, 259]
# Community 5: Nodes [6, 86, 93, 164, 208]
# Community 3: Nodes [7, 123, 130, 172, 270]
# Community 6: Nodes [19, 20, 45, 95, 113, 119, 120, 156, 184, 188, 212, 220, 241, 255]
# Community 1: Nodes [23, 29, 30, 36, 37, 42, 44, 47, 48, 51, 53, 55, 59, 77, 79, 81, 96, 109, 128, 132, 143, 157, 159, 169, 183, 192, 202, 224, 238, 243, 247, 249]
# Community 0: Nodes [43, 66, 85, 112, 176, 203, 266, 267, 268]