# import networkx as nx
# import matplotlib.pyplot as plt  # 这是为了确保能够计算聚集系数

# def calculate_graph_metrics(gexf_file):
#     # 读取GEXF文件
#     G = nx.read_gexf(gexf_file)

#     # 计算基础图属性指标
#     node_count = G.number_of_nodes()
#     edge_count = G.number_of_edges()
#     average_degree = sum(dict(G.degree()).values()) / float(node_count)
#     #clustering_coefficient = nx.average_clustering(G)
#     assortativity = nx.degree_assortativity_coefficient(G)
#     try:
#         diameter = nx.diameter(G)
#     except:
#         diameter = '无限大'  # 对于非连通图，图直径无定义
#     else:
#         diameter = format(diameter, '.2f')  # 如果直径可以计算，保留两位小数
#     density = nx.density(G)
#     average_in_degree = sum(dict(G.in_degree()).values()) / float(node_count)
#     average_out_degree = sum(dict(G.out_degree()).values()) / float(node_count)

#     # 格式化输出，保留两位小数
#     print(f"文件: {gexf_file}")
#     print(f"节点数: {node_count}")
#     print(f"边数量: {edge_count}")
#     print(f"平均度: {average_degree:.2f}")
#     #print(f"聚集系数: {clustering_coefficient:.2f}")
#     print(f"同配系数: {assortativity:.2f}")
#     print(f"图直径: {diameter}")
#     print(f"图密度: {density:.3f}")
#     print(f"平均出度: {average_out_degree:.2f}")
#     print(f"平均入度: {average_in_degree:.2f}")
#     print("-" * 40)

# # 定义GEXF文件列表
# graph_files = [
#     '/home/ta/gambling/www-submission/case_dataset/Arbitrum case dataset/graph_0xc4a482146c2b493066aa7427d23bea4f66e5279c.gexf',
#     '/home/ta/gambling/www-submission/case_dataset/Ethereum case dataset/0xc2a81eb482cb4677136d8812cc6db6e0cb580883.gexf',
#     '/home/ta/gambling/www-submission/case_dataset/Tron case dataset/Tron_hanhua_graph_1.gexf'
# ]

# # 分析每个图文件
# for gexf_file in graph_files:
#     calculate_graph_metrics(gexf_file)

# 给定一gexf图文件，获取边“block_number”属性的全网最大值和最小值

import networkx as nx

def find_min_max_block_number(gexf_path):
    # Load the graph from GEXF file
    G = nx.read_gexf(gexf_path)

    # Initialize variables to store the max and min block numbers
    max_block_number = float('-inf')  # Start with the smallest possible float
    min_block_number = float('inf')   # Start with the largest possible float

    # Iterate over all edges to find the max and min block numbers
    for u, v, data in G.edges(data=True):
        block_number = data.get('blockNum')  # Extract the block number from edge data
        if block_number is not None:  # Ensure that the block number is not missing
            max_block_number = max(max_block_number, block_number)
            min_block_number = min(min_block_number, block_number)

    return min_block_number, max_block_number

# Specify the path to your GEXF file
gexf_path = '/home/ta/gambling/www-submission/case_dataset/Tron case dataset/Tron_hanhua_graph_1.gexf'

# Get the min and max block numbers
min_block_number, max_block_number = find_min_max_block_number(gexf_path)
print(f"Minimum Block Number: {min_block_number}")
print(f"Maximum Block Number: {max_block_number}")
