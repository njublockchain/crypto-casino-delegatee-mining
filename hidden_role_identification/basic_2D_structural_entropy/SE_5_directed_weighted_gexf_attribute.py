import networkx as nx
import math
from networkx.algorithms import cuts
from itertools import chain
import copy
from sklearn.cluster import AgglomerativeClustering

class SE:
    def __init__(self, graph):
        '''
        使用GraphML文件中的图结构进行初始化。
        graph是从GraphML文件读取的networkx.DiGraph对象。
        '''
        self.graph = graph
        self.vol = self.get_g_vol()
        self.division = {}  # {comm1: [node11, node12, ...], comm2: [node21, node22, ...], ...}
        self.struc_data = {}  # 用于存储社区的结构信息
        self.struc_data_2d = {}  # 用于存储社区合并后的结构信息

    def get_g_vol(self):
        return cuts.volume(self.graph, self.graph.nodes, weight='transfer_value')

    def get_cut(self, comm):
        comm_set = {n for n in comm if n in self.graph}
        all_in_edges = self.graph.in_edges(nbunch=comm_set, data=True)
        cut_in_edges = (e for e in all_in_edges if e[0] not in comm_set)
        return sum(data.get('transfer_value', 0) for u, v, data in cut_in_edges)

    def get_volume(self, comm):
        in_degrees = self.graph.in_degree(nbunch=comm, weight='transfer_value')
        return sum(weight for node, weight in in_degrees)

    def calc_1dSE(self):
        SE = 0
        for n in self.graph.nodes:
            d = self.graph.in_degree(n, weight='transfer_value')
            if d > 0:
                SE += - (d / self.vol) * math.log2(d / self.vol)
        return SE

    def calc_2dSE(self):
        SE = 0
        for comm in self.division.values():
            g = self.get_cut(comm)
            v = self.get_volume(comm)
            if v > 0 and self.vol > 0:
                SE += - (g / self.vol) * math.log2(v / self.vol)
            for node in comm:
                d = self.graph.in_degree(node, weight='transfer_value')
                if d > 0 and v > 0:
                    SE += - (d / self.vol) * math.log2(d / v)
        return SE

    def init_division(self):
        ''' 初始化每个节点为一个社区 '''
        self.division = {node: [node] for node in self.graph.nodes}
        for node in self.graph.nodes:
            self.graph.nodes[node]['comm'] = node

    def update_struc_data(self):
        ''' 更新每个社区的结构数据 '''
        for vname in self.division.keys():
            comm = self.division[vname]
            volume = self.get_volume(comm)
            cut = self.get_cut(comm)
            neighbor_comms = []
            for node in comm:
                for k, v in self.division.items():
                    if k != vname and k not in neighbor_comms:
                        for end_node in v:
                            if self.graph.has_edge(node, end_node):
                                neighbor_comms.append(k)
                                break
                        for start_node in v:
                            if self.graph.has_edge(start_node, node):
                                neighbor_comms.append(k)
                                break
            neighbor_comms = list(set(neighbor_comms))
            vSE = - (cut / self.vol) * math.log2(volume / self.vol) if volume != 0 else 0
            vnodeSE = 0
            for node in comm:
                d = self.graph.in_degree(node, weight='transfer_value')
                if d != 0:
                    vnodeSE -= (d / self.vol) * math.log2(d / volume)
            self.struc_data[vname] = [volume, cut, vSE, vnodeSE, neighbor_comms]

    def update_struc_data_2d(self):
        ''' 更新合并后的社区结构数据 '''
        all_comms = list(self.division.keys())
        all_comms.sort()
        for v1 in all_comms:
            neighbor_comms = self.struc_data[v1][4]
            for v2 in neighbor_comms:
                if v1 < v2:
                    k = (v1, v2)
                    comm_merged = self.division[v1] + self.division[v2]
                    gm = self.get_cut(comm_merged)
                    vm = self.struc_data[v1][0] + self.struc_data[v2][0]
                    if self.struc_data[v1][0] == 0 or self.struc_data[v2][0] == 0:
                        vmSE = self.struc_data[v1][2] + self.struc_data[v2][2]
                        vmnodeSE = self.struc_data[v1][3] + self.struc_data[v2][3]
                    else:
                        vmSE = - (gm / self.vol) * math.log2(vm / self.vol)
                        vmnodeSE = self.struc_data[v1][3] - (self.struc_data[v1][0] / self.vol) * math.log2(self.struc_data[v1][0] / vm) + \
                                   self.struc_data[v2][3] - (self.struc_data[v2][0] / self.vol) * math.log2(self.struc_data[v2][0] / vm)
                    self.struc_data_2d[k] = [vm, gm, vmSE, vmnodeSE]

    def update_division_MinSE(self):
        ''' 进行结构熵最小化 '''
        def Mg_operator(v1, v2):
            v1SE = self.struc_data[v1][2]
            v1nodeSE = self.struc_data[v1][3]
            v2SE = self.struc_data[v2][2]
            v2nodeSE = self.struc_data[v2][3]
            k = (v1, v2)
            vm, gm, vmSE, vmnodeSE = self.struc_data_2d[k]
            delta_SE = vmSE + vmnodeSE - (v1SE + v1nodeSE + v2SE + v2nodeSE)
            return delta_SE

        while True:
            delta_SE = 99999
            vm1 = None
            vm2 = None
            all_comms = list(self.division.keys())
            all_comms.sort()
            for v1 in all_comms:
                neighbor_comms = self.struc_data[v1][4]
                for v2 in neighbor_comms:
                    if v1 < v2:
                        new_delta_SE = Mg_operator(v1, v2)
                        if new_delta_SE < delta_SE:
                            delta_SE = new_delta_SE
                            vm1 = v1
                            vm2 = v2

            if delta_SE < 0:
                for node in self.division[vm2]:
                    self.graph.nodes[node]['comm'] = vm1
                self.division[vm1] += self.division[vm2]
                self.division.pop(vm2)
                volume = self.struc_data[vm1][0] + self.struc_data[vm2][0]
                cut = self.get_cut(self.division[vm1])
                neighbor_comms = set(self.struc_data[vm1][4] + self.struc_data[vm2][4])
                neighbor_comms.remove(vm2)
                neighbor_comms = list(neighbor_comms)
                vmSE = - (cut / self.vol) * math.log2(volume / self.vol)
                vmnodeSE = self.struc_data[vm1][3] - (self.struc_data[vm1][0] / self.vol) * math.log2(self.struc_data[vm1][0] / volume) + \
                           self.struc_data[vm2][3] - (self.struc_data[vm2][0] / self.vol) * math.log2(self.struc_data[vm2][0] / volume)
                self.struc_data[vm1] = [volume, cut, vmSE, vmnodeSE, neighbor_comms]
                vm2_neighbors = self.struc_data[vm2][4]
                vm2_neighbors.remove(vm1)
                if vm2 in vm2_neighbors:
                    vm2_neighbors.remove(vm2)
                for node in vm2_neighbors:
                    node_neighbors = self.struc_data[node][4]
                    node_neighbors.remove(vm2)
                    node_neighbors.append(vm1)
                    self.struc_data[node][4] = list(set(node_neighbors))
                self.struc_data.pop(vm2)
                struc_data_2d_new = {}
                for k in self.struc_data_2d.keys():
                    if k[0] == vm2 or k[1] == vm2:
                        v = [k[0], k[1], vm1]
                        v.remove(vm2)
                        v = list(set(v))
                        if len(v) < 2:
                            continue
                        v.sort()
                        v1 = v[0]
                        v2 = v[1]
                        comm_merged = self.division[v1] + self.division[v2]
                        gm = self.get_cut(comm_merged)
                        vm = self.struc_data[v1][0] + self.struc_data[v2][0]
                        if self.struc_data[v1][0] == 0 or self.struc_data[v2][0] == 0:
                            vmSE = self.struc_data[v1][2] + self.struc_data[v2][2]
                            vmnodeSE = self.struc_data[v1][3] + self.struc_data[v2][3]
                        else:
                            vmSE = - (gm / self.vol) * math.log2(vm / self.vol)
                            vmnodeSE = self.struc_data[v1][3] - (self.struc_data[v1][0] / self.vol) * math.log2(self.struc_data[v1][0] / vm) + \
                                       self.struc_data[v2][3] - (self.struc_data[v2][0] / self.vol) * math.log2(self.struc_data[v2][0] / vm)
                        struc_data_2d_new[(v1, v2)] = [vm, gm, vmSE, vmnodeSE]
                    elif k[0] == vm1 or k[1] == vm1:
                        v1 = k[0]
                        v2 = k[1]
                        comm_merged = self.division[v1] + self.division[v2]
                        gm = self.get_cut(comm_merged)
                        vm = self.struc_data[v1][0] + self.struc_data[v2][0]
                        if self.struc_data[v1][0] == 0 or self.struc_data[v2][0] == 0:
                            vmSE = self.struc_data[v1][2] + self.struc_data[v2][2]
                            vmnodeSE = self.struc_data[v1][3] + self.struc_data[v2][3]
                        else:
                            vmSE = - (gm / self.vol) * math.log2(vm / self.vol)
                            vmnodeSE = self.struc_data[v1][3] - (self.struc_data[v1][0] / self.vol) * math.log2(self.struc_data[v1][0] / vm) + \
                                       self.struc_data[v2][3] - (self.struc_data[v2][0] / self.vol) * math.log2(self.struc_data[v2][0] / vm)
                        struc_data_2d_new[k] = [vm, gm, vmSE, vmnodeSE]
                    else:
                        struc_data_2d_new[k] = self.struc_data_2d[k]
                self.struc_data_2d = struc_data_2d_new
            else:
                break

def extract_embeddings(graph):
    '''提取每个节点的embedding特征，返回一个字典，节点id为键，embedding为值'''
    embeddings = {}
    for node, data in graph.nodes(data=True):
        embedding = [data.get(f'embedding_{i}', 0.0) for i in range(54)]  # 提取embedding_0到embedding_19
        embeddings[node] = embedding
    return embeddings

def ahc_clustering(graph, n_clusters):
    '''使用Agglomerative Hierarchical Clustering (AHC)进行聚类'''
    embeddings = extract_embeddings(graph)
    X = list(embeddings.values())
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)

    # 更新图节点的社区划分信息
    for node, label in zip(graph.nodes, labels):
        graph.nodes[node]['comm'] = label

    # 生成初始的社区划分字典
    division = {}
    for node, label in zip(graph.nodes, labels):
        if label not in division:
            division[label] = []
        division[label].append(node)

    return division

def calculate_2d_se_for_clusters(graph, min_clusters=2, max_clusters=10):
    '''计算不同社区数量下的2D结构熵，并返回最优社区划分结果'''
    results = []
    best_se = float('inf')  # 用于记录最小的2D结构熵
    best_division = None    # 用于记录最优的社区划分

    for n_clusters in range(min_clusters, max_clusters + 1):
        print(f'Calculating for {n_clusters} clusters...')
        division = ahc_clustering(graph, n_clusters)  # 调用聚类算法
        seg = SE(graph)
        seg.division = division
        seg.update_struc_data()
        seg.update_struc_data_2d()
        SE_2d = seg.calc_2dSE()
        results.append((n_clusters, SE_2d, division))

        # 记录结构熵最小的社区划分
        if SE_2d < best_se:
            best_se = SE_2d
            best_division = division

    return results, best_division

def test_SE():
    '''测试2D结构熵计算并展示最优社区划分结果'''
    graph = nx.read_graphml("/home/lab0/gnn-wjx/0xc2a81eb482cb4677136d8812cc6db6e0cb580883_embedding_with_tef.graphml")

    # 计算不同社区数量下的2D结构熵，并获得最优社区划分
    results, best_division = calculate_2d_se_for_clusters(graph, 2, 10)
    
    # 打印不同社区数量下的结构熵
    for n_clusters, SE_2d, _ in results:
        print(f'Number of communities: {n_clusters}, 2D SE: {SE_2d}')
    
    # 打印最优社区划分
    print("\nBest community division (minimal SE):")
    for comm_id, nodes in best_division.items():
        print(f'Community {comm_id}: {nodes}')

if __name__ == "__main__":
    test_SE()