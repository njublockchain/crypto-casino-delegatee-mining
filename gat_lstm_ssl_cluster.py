import torch
import torch.nn as nn
import hdbscan
import torch.optim as optim
from torch.nn import Sequential
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

# 假设graphml文件路径为'path_to_graph.graphml'，并且我们知道节点属性的列名。
graphml_path = '0xc2a81eb482cb4677136d8812cc6db6e0cb580883_complete.graphml'

# 读取GraphML文件
G = nx.read_graphml(graphml_path)

# 定义所有节点应该具有的属性及其默认值
default_attributes = {'_nx_name': 0, 'stop': 0, 'add': 0, 'mul': 0, 'sub': 0, 'div': 0, 'sdiv': 0, 'mod': 0, 'smod': 0, 'addmod': 0, 'mulmod': 0, 'exp': 0, 'signextend': 0, 'lt': 0, 'gt': 0, 'slt': 0, 'sgt': 0, 'eq': 0, 'iszero': 0, 'and': 0, 'or': 0, 'xor': 0, 'not': 0, 'byte': 0, 'shl': 0, 'shr': 0, 'sar': 0, 'sha3': 0, 'address': 0, 'balance': 0, 'origin': 0, 'caller': 0, 'callvalue': 0, 'calldataload': 0, 'calldatasize': 0, 'calldatacopy': 0, 'codesize': 0, 'codecopy': 0, 'gasprice': 0, 'extcodesize': 0, 'extcodecopy': 0, 'returndatasize': 0, 'returndatacopy': 0, 'extcodehash': 0, 'blockhash': 0, 'coinbase': 0, 'timestamp': 0, 'number': 0, 'difficulty': 0, 'gaslimit': 0, 'chainid': 0, 'selfbalance': 0, 'pop': 0, 'mload': 0, 'mstore': 0, 'mstore8': 0, 'sload': 0, 'sstore': 0, 'jump': 0, 'jumpi': 0, 'getpc': 0, 'msize': 0, 'gas': 0, 'jumpdest': 0, 'push1': 0, 'push2': 0, 'push3': 0, 'push4': 0, 'push5': 0, 'push6': 0, 'push7': 0, 'push8': 0, 'push9': 0, 'push10': 0, 'push11': 0, 'push12': 0, 'push13': 0, 'push14': 0, 'push15': 0, 'push16': 0, 'push17': 0, 'push18': 0, 'push19': 0, 'push20': 0, 'push21': 0, 'push22': 0, 'push23': 0, 'push24': 0, 'push25': 0, 'push26': 0, 'push27': 0, 'push28': 0, 'push29': 0, 'push30': 0, 'push31': 0, 'push32': 0, 'dup1': 0, 'dup2': 0, 'dup3': 0, 'dup4': 0, 'dup5': 0, 'dup6': 0, 'dup7': 0, 'dup8': 0, 'dup9': 0, 'dup10': 0, 'dup11': 0, 'dup12': 0, 'dup13': 0, 'dup14': 0, 'dup15': 0, 'dup16': 0, 'swap1': 0, 'swap2': 0, 'swap3': 0, 'swap4': 0, 'swap5': 0, 'swap6': 0, 'swap7': 0, 'swap8': 0, 'swap9': 0, 'swap10': 0, 'swap11': 0, 'swap12': 0, 'swap13': 0, 'swap14': 0, 'swap15': 0, 'swap16': 0, 'log0': 0, 'log1': 0, 'log2': 0, 'log3': 0, 'log4': 0, 'create': 0, 'call': 0, 'callcode': 0, 'return': 0, 'delegatecall': 0, 'create2': 0, 'staticcall': 0, 'revert': 0, 'invalid': 0, 'selfdestruct':0,
    'sum_val_in': 0, 'sum_val_out': 0, 'avg_val_in': 0, 'avg_val_out': 0,
    'count_in': 0, 'count_out': 0, 'count': 0, 'freq': 0,
    'freq_in': 0, 'freq_out': 0, 'gini_val': 0, 'gini_val_in': 0,
    'gini_val_out': 0, 'avg_gas': 0, 'avg_gas_in': 0, 'avg_gas_out': 0,
    'avg_gasprice': 0, 'avg_gasprice_in': 0, 'avg_gasprice_out': 0,
    'in_out_rate': 0
}
# 遍历图中的所有节点，确保每个节点都具有这些属性
for node in G.nodes():
    for attr, default_val in default_attributes.items():
        G.nodes[node].setdefault(attr, default_val)

# Define default values for edge attributes
default_edge_attrs = {
    'gas_used': 0.0,
    'edge_weight': 0.0,
    'block_number': 0.0,
    'effective_gas_price': 0.0,
    'transfer_value': 0.0
}

# Ensure every edge has the required attributes
for u, v, attrs in G.edges(data=True):
    for key, default_value in default_edge_attrs.items():
        attrs.setdefault(key, default_value)

# 将NetworkX图转换为PyTorch Geometric图数据
data = from_networkx(G)

# 为每个节点创建初始特征向量
#node_features = torch.tensor([data.node[n]['feature'] for n in range(data.num_nodes)], dtype=torch.float)

# 为每条边创建边特征
# Extracting edge features
edge_features = torch.tensor([[attrs['transfer_value'], attrs['block_number']]
                              for _, _, attrs in G.edges(data=True)], dtype=torch.float)

# 定义GAT模型
class CustomGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomGATConv, self).__init__(node_dim=0)  # 基于节点的消息传递
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)  # 输入特征到输出特征的线性变换
        self.att_lin = torch.nn.Linear(2 * out_channels, 1, bias=False)  # 注意力机制的线性变换

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

    def forward(self, node_features, structural_features, temporal_features):
        concatenated_features = torch.cat((node_features, structural_features, temporal_features), dim=1)
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

node_feature_keys = ['sum_val_in', 'sum_val_out', 'avg_val_in', 'avg_val_out',
    'count_in', 'count_out', 'count', 'freq',
    'freq_in', 'freq_out', 'gini_val', 'gini_val_in',
    'gini_val_out', 'avg_gas', 'avg_gas_in', 'avg_gas_out',
    'avg_gasprice', 'avg_gasprice_in', 'avg_gasprice_out',
    'in_out_rate']

# Extract node features
node_features = []
for _, node_data in G.nodes(data=True):
    node_features.append([node_data.get(key, 0) for key in node_feature_keys])  # Default to 0 if key doesn't exist

# Convert node features to a tensor
node_features = torch.tensor(node_features, dtype=torch.float)

# 提取每个节点的block_number序列
node_to_block_numbers = defaultdict(list)
for u, v, edge_attr in G.edges(data=True):
    node_to_block_numbers[u].append(edge_attr['block_number'])
    node_to_block_numbers[v].append(edge_attr['block_number'])

# 为每个节点的block_number序列进行填充或截断，以确保统一的序列长度
max_seq_length = 5  # 你可以根据需要调整这个值
block_number_sequences = []
for blocks in node_to_block_numbers.values():
    padded_blocks = blocks[:max_seq_length] + [0] * (max_seq_length - len(blocks))
    block_number_sequences.append(padded_blocks)

# 转换为张量
block_number_sequences_tensor = torch.tensor(block_number_sequences, dtype=torch.float).unsqueeze(-1)

# 构建GAT模型来获取结构特征
edge_weights = torch.tensor([attrs['transfer_value'] for _, _, attrs in G.edges(data=True)], dtype=torch.float)
gat = GATNet(node_features.shape[1], 8)
structural_features = gat(node_features, data.edge_index, edge_weights)

# 使用LSTM处理时间序列
lstm_model = TransactionLSTM(1, 64)  # 假设每个时间步只有1个特征(block_number)
temporal_features = lstm_model(block_number_sequences_tensor)

# 融合特征
# 假设你已经有了结构特征和时间特征，下面是融合特征的代码
# feature_dims = [structural_features.shape[1], temporal_features.shape[1]]  # 更新特征维

gat_loss_fn = nn.MSELoss()  # or any appropriate loss function
lstm_loss_fn = nn.MSELoss()  # or any appropriate loss function
self_supervised_loss_fn = SelfSupervisedLoss()


print("Node Features Shape:", node_features.shape)
print("Structural Features Shape:", structural_features.shape)
print("Temporal Features Shape:", temporal_features.shape)


# 融合三类特征
# concatenated_features = torch.cat((structural_features, temporal_features), dim=1)
# self_attention_fusion_net = SelfAttentionFusionNet(concatenated_features.shape[1])
# final_representation = self_attention_fusion_net(concatenated_features)
concatenated_features = torch.cat((node_features, structural_features, temporal_features), dim=1)
final_representation = concatenated_features

# feature_dims = [8, 64]  # The sizes of each feature set
# self_attention_aggregation = SelfAttentionAggregation(feature_dims)
#
# final_representation = self_attention_aggregation(structural_features, temporal_features)

print(final_representation)

if final_representation.dim() == 1:
    # Reshape it to be 2D ([num_nodes, num_features])
    final_representation = final_representation.view(-1, 1)

final_representation_detached = final_representation.detach().cpu().numpy()

# # KMeans聚类
# kmeans_clusterer = KMeans(n_clusters=11, random_state=0)
# cluster_labels_kmeans = kmeans_clusterer.fit_predict(final_representation_detached)

# # 计算KMeans聚类的轮廓系数
# silhouette_coeff_kmeans = silhouette_score(final_representation_detached, cluster_labels_kmeans)
# print(f"Silhouette Coefficient (KMeans): {silhouette_coeff_kmeans}")

# # AgglomerativeClustering聚类
# ahc_clusterer = AgglomerativeClustering(n_clusters=11)
# cluster_labels_ahc = ahc_clusterer.fit_predict(final_representation_detached)

# # 计算AgglomerativeClustering聚类的轮廓系数
# silhouette_coeff_ahc = silhouette_score(final_representation_detached, cluster_labels_ahc)
# print(f"Silhouette Coefficient (AgglomerativeClustering): {silhouette_coeff_ahc}")

# 使用HDBSCAN进行聚类
hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=11, gen_min_span_tree=True)

# 假设final_representation_detached是你的节点嵌入/特征
cluster_labels = hdbscan_clusterer.fit_predict(final_representation_detached)

# 计算轮廓系数，需要注意的是轮廓系数不能在存在噪声点（即标签为-1的点）时计算
# 所以我们只在非噪声点上计算轮廓系数
mask = cluster_labels != -1
if np.sum(mask) > 1:  # 至少要有两个非噪声点才能计算轮廓系数
    silhouette_coeff = silhouette_score(final_representation_detached[mask], cluster_labels[mask])
    print(f"Silhouette Coefficient: {silhouette_coeff}")
else:
    print("Not enough clusters to calculate the silhouette coefficient.")

# HDBSCAN不一定会为所有点分配簇，它可能会将一些点标记为噪声，用-1表示
unique_clusters = set(cluster_labels)
print(f"Unique clusters found by HDBSCAN: {unique_clusters}")

# 显示每一类中所有节点的指定属性值
attribute_name = 'addr'  # 指定想要查看的属性
# 创建从索引到节点ID的映射
index_to_node_id = list(G.nodes())
clusters = defaultdict(list)
for i, label in enumerate(cluster_labels):
    node_id = index_to_node_id[i]  # 通过索引获取节点ID
    attribute_value = G.nodes[node_id].get(attribute_name, None)  # 获取节点的属性值
    clusters[label].append(attribute_value)

for cluster, attribute_values in clusters.items():
    print(f"Cluster {cluster}: {attribute_name} values {attribute_values}")

# 为每个节点设置HDBSCAN聚类标签作为属性
for i, node_id in enumerate(G.nodes()):
    G.nodes[node_id]['kdd_cluster'] = int(cluster_labels[i])

# 将聚类结果写入GEXF文件
nx.write_gexf(G, "updated_graph_with_hdbscan.gexf")
