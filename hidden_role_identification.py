from sklearnex import patch_sklearn
patch_sklearn()
import torch
import torch.nn as nn
import hdbscan
import pickle
from torch_geometric.data import Data
import torch.optim as optim
from torch.nn import Sequential
from sklearn.preprocessing import normalize
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
from torch_geometric.nn import GATConv
from torch_geometric.utils.convert import from_networkx
from sklearn.cluster import KMeans
import networkx as nx
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import softmax
from collections import defaultdict
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering


# 假设pkl文件路径
pkl_path = '/home/ta/gambling/www-submission/case_dataset/Ethereum case dataset/0xc2a81eb482cb4677136d8812cc6db6e0cb580883.pkl'


# 读取pkl文件
with open(pkl_path, 'rb') as f:
    G = pickle.load(f)

# 创建节点ID到连续整数索引的映射
node_id_to_index = {node_id: idx for idx, node_id in enumerate(G.nodes())}

# 使用映射来构建edge_index
edge_index = torch.tensor(
    [[node_id_to_index[u], node_id_to_index[v]] for u, v in G.edges()], 
    dtype=torch.long
).t().contiguous()

# 定义节点特征的标题列表
node_feature_titles = [
    "degree", "degree_in", "degree_out", "degree_diff_ratio", "degree_io_ratio",
    "value_in_sum", "value_out_sum", "value_in_avg", "value_out_avg", "value_sum",
    "value_diff_ratio", "value_io_ratio", "blocknum_in_avg", "blocknum_out_avg",
    "blocknum_avg", "blocknum_diff_ratio", "usdt_transfer_in", "usdt_transfer_out",
    "usdt_transfer_all", "usdt_transfer_diff_ratio", "usdt_transfer_io_ratio",
    "account_transaction_count_out", "account_transaction_count_in", "account_transaction_count_all",
    "account_transaction_diff_ratio", "account_transaction_io_ratio"
]

# 提取节点特征
node_features = []
for node in G.nodes():
    node_attrs = G.nodes[node]
    features = []
    for title in node_feature_titles:
        # 对于未定义的属性，使用0作为默认值
        features.append(node_attrs.get(title, 0))
    node_features.append(features)

# 对节点特征进行归一化
node_features_nor = normalize(node_features, axis=0)  # 归一化每个特征维度

# 将列表转换为张量
node_features_tensor = torch.tensor(node_features_nor, dtype=torch.float)

# 提取边属性
edge_attributes = []
for u, v, attrs in G.edges(data=True):
    edge_attributes.append([attrs.get('value', 0), attrs.get('blockNum', 0)])

# 将边属性列表转换为张量
edge_attr = torch.tensor(edge_attributes, dtype=torch.float)


# 创建Data对象
data = Data(x=node_features_tensor, edge_index=edge_index, edge_attr=edge_attr)

# 使用归一化后的节点特征
x = node_features_tensor

# 定义GAT模型
class CustomGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.5):
        super(CustomGATConv, self).__init__(node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        
        # Assume out_channels is total dimension, each head has out_channels // heads dimension
        self.lin = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.att_lin = nn.Linear(2 * (out_channels // heads), 1, bias=False)
        
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.att_lin.weight)

    def forward(self, x, edge_index, edge_weight):
        x = self.lin(x)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_weight=edge_weight)

    def message(self, edge_index_i, x_i, x_j, edge_weight):
        # 计算注意力系数
        x_cat = torch.cat([x_i, x_j], dim=-1)
        alpha = self.att_lin(x_cat)
        alpha = F.leaky_relu(alpha, 0.2)

        # Incorporate edge weights into attention coefficients
        alpha = alpha.squeeze(-1) * edge_weight

        alpha = softmax(alpha, edge_index_i)
        return x_j * alpha.view(-1, 1)

    def update(self, aggr_out):
        # 更新节点嵌入
        return aggr_out


# Update your GATNet class to use CustomGATConv
class GATNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(GATNet, self).__init__()
        self.conv1 = CustomGATConv(in_features, out_features)

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        return x

# 定义LSTM模型
# 修改LSTM模型以适应新的输入
class TransactionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TransactionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x):
        # LSTM需要的输入形状为(batch, seq_len, features)
        # 我们已经有了正确的形状，直接进行前向传播
        out, (hn, cn) = self.lstm(x)
        # 选择最后一个时间步的隐藏状态
        return hn[-1]

# 定义自注意力融合模型
class SelfAttentionAggregation(nn.Module):
    def __init__(self, feature_dims):
        super(SelfAttentionAggregation, self).__init__()
        self.feature_dims = feature_dims
        self.attention_weights = nn.Parameter(torch.Tensor(1, sum(feature_dims)))
        nn.init.xavier_uniform_(self.attention_weights)

    def forward(self, node_features_tensor, structural_features, temporal_features):
        concatenated_features = torch.cat((node_features_tensor, structural_features, temporal_features), dim=1)
        attention_scores = F.softmax(torch.matmul(concatenated_features, self.attention_weights.T), dim=1)
        attended_features = attention_scores * concatenated_features
        return torch.sum(attended_features, dim=1)

    # def forward(self, structural_features, temporal_features):
    #     concatenated_features = torch.cat((structural_features, temporal_features), dim=1)
    #     attention_scores = F.softmax(torch.matmul(concatenated_features, self.attention_weights.T), dim=1)
    #     attended_features = attention_scores * concatenated_features
    #     return torch.sum(attended_features, dim=1)

class SelfSupervisedLoss(nn.Module):
    def __init__(self):
        super(SelfSupervisedLoss, self).__init__()

    def forward(self, embeddings, cluster_labels):
        # Assuming cluster_labels are obtained via some clustering applied on embeddings
        unique_labels = torch.unique(cluster_labels)
        loss = 0
        for label in unique_labels:
            cluster_indices = (cluster_labels == label).nonzero(as_tuple=True)[0]
            if len(cluster_indices) > 1:  # Ensure there are at least two nodes in the cluster to compute distance
                cluster_embeddings = embeddings[cluster_indices]
                # Normalize embeddings to unit norm for stable distance computation
                cluster_embeddings = F.normalize(cluster_embeddings, p=2, dim=1)
                # Compute pairwise distances within the cluster
                distances = pairwise_distances(cluster_embeddings.cpu().detach().numpy(), metric='euclidean')
                # Sum of distances within the cluster
                cluster_loss = torch.sum(torch.tensor(distances))
                loss += cluster_loss
        loss = loss / len(unique_labels)  # Average loss across clusters
        print("loss", loss)
        return loss

class CombinedModel(nn.Module):
    def __init__(self, gat_model, lstm_model):
        super(CombinedModel, self).__init__()
        self.gat_model = gat_model
        self.lstm_model = lstm_model

    def forward(self, x, edge_index, edge_attr, temporal_features):
        gat_out = self.gat_model(x, edge_index, edge_attr)
        lstm_out = self.lstm_model(temporal_features)
        return gat_out, lstm_out


# 提取每个节点的block_number序列
node_to_block_numbers = defaultdict(list)
for u, v, edge_attr in G.edges(data=True):
    node_to_block_numbers[u].append(edge_attr['blockNum'])
    node_to_block_numbers[v].append(edge_attr['blockNum'])

# 为每个节点的block_number序列进行填充或截断，以确保统一的序列长度
max_seq_length = 5  # 你可以根据需要调整这个值
block_number_sequences = []
for blocks in node_to_block_numbers.values():
    padded_blocks = blocks[:max_seq_length] + [0] * (max_seq_length - len(blocks))
    block_number_sequences.append(padded_blocks)

# 转换为张量
block_number_sequences_tensor = torch.tensor(block_number_sequences, dtype=torch.float).unsqueeze(-1)

learning_rate = 0.001
epochs = 100
batch_size = 32
gamma = 0.1

# 构建GAT模型来获取结构特征
edge_weights = torch.tensor([attrs['value'] for _, _, attrs in G.edges(data=True)], dtype=torch.float)
gat = GATNet(node_features_tensor.shape[1], 8)
structural_features = gat(node_features_tensor, data.edge_index, edge_weights)

# 使用LSTM处理时间序列
lstm_model = TransactionLSTM(1, 64)  # 假设每个时间步只有1个特征(block_number)
temporal_features = lstm_model(block_number_sequences_tensor)

# 融合特征
# 假设你已经有了结构特征和时间特征，下面是融合特征的代码
# feature_dims = [structural_features.shape[1], temporal_features.shape[1]]  # 更新特征维

gat_loss_fn = nn.MSELoss()  # or any appropriate loss function
lstm_loss_fn = nn.MSELoss()  # or any appropriate loss function
self_supervised_loss_fn = SelfSupervisedLoss()

# optimizer = optim.Adam(gat.parameters(), lr=learning_rate)
# criterion = gat_loss_fn + gamma * self_supervised_loss_fn


print("Node Features Shape:", node_features_tensor.shape)
print("Structural Features Shape:", structural_features.shape)
print("Temporal Features Shape:", temporal_features.shape)


# 融合三类特征
# concatenated_features = torch.cat((structural_features, temporal_features), dim=1)
# self_attention_fusion_net = SelfAttentionFusionNet(concatenated_features.shape[1])
# final_representation = self_attention_fusion_net(concatenated_features)
concatenated_features = torch.cat((node_features_tensor, structural_features, temporal_features), dim=1)
final_representation = concatenated_features


print(final_representation)

if final_representation.dim() == 1:
    # Reshape it to be 2D ([num_nodes, num_features])
    final_representation = final_representation.view(-1, 1)

final_representation_detached = final_representation.detach().cpu().numpy()

output_data = {
    'final_representation': final_representation_detached,  # 假设这是你从模型中得到的节点表示
    'index_to_node_id': {idx: node_id for node_id, idx in node_id_to_index.items()}  # 索引到节点ID的映射
}

with open('/home/lab0/coinwar/key_node/updated_graph_with_for_critical_node.pkl', 'wb') as f:
    pickle.dump(output_data, f)

# # 假设final_representation_detached是你的节点嵌入/特征
# max_clusters = 20  # 设置最大可能的簇数
# best_silhouette_score = -1
# best_cluster_labels = None
# best_min_cluster_size = 50  # 初始值

# for min_cluster_size in range(2, 50):  # 尝试不同的min_cluster_size值
#     hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)
#     cluster_labels = hdbscan_clusterer.fit_predict(final_representation_detached)

#     # 计算轮廓系数，需要注意的是轮廓系数不能在存在噪声点（即标签为-1的点）时计算
#     # 所以我们只在非噪声点上计算轮廓系数
#     mask = cluster_labels != -1
#     if np.sum(mask) > 1:  # 至少要有两个非噪声点才能计算轮廓系数
#         silhouette_coeff = silhouette_score(final_representation_detached[mask], cluster_labels[mask])
#         print(f"Silhouette Coefficient for min_cluster_size={min_cluster_size}: {silhouette_coeff}")

#         # 如果当前轮廓系数更好，则更新最佳结果
#         if silhouette_coeff > best_silhouette_score:
#             best_silhouette_score = silhouette_coeff
#             best_cluster_labels = cluster_labels
#             best_min_cluster_size = min_cluster_size

# # 使用具有最大轮廓系数的聚类结果和对应的min_cluster_size
# cluster_labels = best_cluster_labels
# min_cluster_size = best_min_cluster_size

# index_to_node_id = list(G.nodes())
# # 显示每一类中所有节点的ID
# clusters = defaultdict(list)
# for i, label in enumerate(cluster_labels):
#     node_id = index_to_node_id[i]  # 通过索引获取节点ID
#     clusters[label].append(node_id)

# for cluster, node_ids in clusters.items():
#     print(f"Cluster {cluster}: Node IDs {node_ids}")

# # 为每个节点设置HDBSCAN聚类标签作为属性
# for i, node_id in enumerate(G.nodes()):
#     G.nodes[node_id]['kdd_cluster'] = int(cluster_labels[i])

# # 将聚类结果写入GEXF文件
# nx.write_gexf(G, "/home/lab0/coinwar/key_node/updated_graph_with_hdbscan.gexf")

# print(f"Best Silhouette Coefficient: {best_silhouette_score}")
# print(cluster_labels)
# print(f"Best min_cluster_size: {min_cluster_size}")


max_clusters = 20  # 设置最大可能的簇数
best_silhouette_score = -1
best_cluster_labels = None

for n_clusters in range(2, max_clusters + 1):
    ahc_clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = ahc_clusterer.fit_predict(final_representation_detached)

    # 计算轮廓系数，需要注意的是轮廓系数不能在存在噪声点（即标签为-1的点）时计算
    # 所以我们只在非噪声点上计算轮廓系数
    mask = cluster_labels != -1
    if np.sum(mask) > 1:  # 至少要有两个非噪声点才能计算轮廓系数
        silhouette_coeff = silhouette_score(final_representation_detached[mask], cluster_labels[mask])
        print(f"Silhouette Coefficient for {n_clusters} clusters: {silhouette_coeff}")

        # 如果当前轮廓系数更好，则更新最佳结果
        if silhouette_coeff > best_silhouette_score:
            best_silhouette_score = silhouette_coeff
            best_cluster_labels = cluster_labels

 # 使用具有最大轮廓系数的聚类结果
cluster_labels = best_cluster_labels
print(f"Best Silhouette Coefficient: {best_silhouette_score}")

index_to_node_id = list(G.nodes())
 # 显示每一类中所有节点的ID
clusters = defaultdict(list)
for i, label in enumerate(cluster_labels):
    node_id = index_to_node_id[i]  # 通过索引获取节点ID
    clusters[label].append(node_id)

for cluster, node_ids in clusters.items():
    print(f"Cluster {cluster}: Node IDs {node_ids}")

 # 为每个节点设置AHC聚类标签作为属性
for i, node_id in enumerate(G.nodes()):
    G.nodes[node_id]['ahc_cluster'] = int(cluster_labels[i])

#  # 将聚类结果写入GEXF文件
# nx.write_gexf(G, "/home/lab0/coinwar/key_node/updated_graph_with_ahc_pruned_not_temporal.gexf")

# # 将图G保存到Pickle文件中
# with open("/home/lab0/coinwar/key_edge/updated_graph_with_clusters.pkl", 'wb') as f:
#     pickle.dump(G, f)

# %%
