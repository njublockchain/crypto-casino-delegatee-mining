import networkx as nx
import numpy as np
import math
import copy
from networkx.algorithms import cuts
from itertools import chain

class SE:
    def __init__(self, A):
        '''
        # A = (a_{i2,i1}), where a_{i2,i1} \in {0, 1} for i2, i1 = 1, ..., n. n is the total number of objects, i.e., nodes. 
        # (i2, i1) stands for ((t-1)-th state, t-th state), i.e., (X_{t-1}, X_t).
        # a_{i2,i1} = 1 indicates that there is an edge starts from i2 and points to i1.
        '''
        self.A = A
        self.graph = nx.stochastic_graph(nx.from_numpy_array(A, create_using=nx.DiGraph), weight='weight')
        self.vol = self.get_g_vol()
        self.division = {}  # {comm1: [node11, node12, ...], comm2: [node21, node22, ...], ...}
        self.struc_data = {}  # {comm1: [vol1, cut1, community_node_SE, leaf_nodes_SE, neighbor_comms1], comm2:[vol2, cut2, community_node_SE, leaf_nodes_SE, neighbor_comms2]，... }
        self.struc_data_2d = {} # {(comm1, comm2): [vol_after_merge, cut_after_merge, comm_node_SE_after_merge, leaf_nodes_SE_after_merge], (comm1, comm3): [], ...}
  
    # 计算图的总容量（volume）
    def get_g_vol(self):
        return cuts.volume(self.graph, self.graph.nodes, weight = 'weight')
  
    # 计算社区的割边总和（cut size）和容量（volume）
    def get_cut(self, comm):
        comm_set = {n for n in comm if n in self.graph}
        # all in_edges to the nodes in comm
        all_in_edges = self.graph.in_edges(nbunch = comm_set, data = 'weight')
        # in_edges from nodes out of comm to the nodes in comm
        cut_in_edges = (e for e in all_in_edges if e[0] not in comm_set)
        return sum(weight for u, v, weight in cut_in_edges)
  
    def get_volume(self, comm):
        in_degrees = self.graph.in_degree(nbunch = comm, weight = 'weight')
        return sum(weight for node, weight in in_degrees)
  
    # 计算图的一维结构熵（1D SE）
    def calc_1dSE(self):
        SE = 0
        for n in self.graph.nodes:
            d = self.graph.in_degree(n, weight = 'weight')
            if d > 0:
                SE += - (d / self.vol) * math.log2(d / self.vol)
        return SE
    
    # 计算图的二维结构熵（2D SE）
    def calc_2dSE(self):
        SE = 0
        for comm in self.division.values():
            g = self.get_cut(comm)
            v = self.get_volume(comm)
            SE += - (g / self.vol) * math.log2(v / self.vol)
            for node in comm:
                d = self.graph.in_degree(node, weight = 'weight')
                SE += - (d / self.vol) * math.log2(d / v)
        return SE
    
    # 更新结构数据
    def update_struc_data(self):
        for vname in self.division.keys():
            comm = self.division[vname]
            volume = self.get_volume(comm)
            cut = self.get_cut(comm)
            neighbor_comms = []
            for node in comm:
                # out neighbors:
                for k,v in self.division.items():
                    if k != vname and k not in neighbor_comms:
                        for end_node in v:
                            if self.A[node, end_node] !=  0:
                                neighbor_comms.append(k)
                                break
                # in neighbors:
                for k,v in self.division.items():
                    if k != vname and k not in neighbor_comms:
                        for start_node in v:
                            if self.A[start_node, node] !=  0:
                                neighbor_comms.append(k)
                                break
            neighbor_comms = list(set(neighbor_comms))

            if volume == 0:
                vSE = 0
            else:
                vSE = - (cut / self.vol) * math.log2(volume / self.vol)
            vnodeSE = 0
            for node in comm:
                d = self.graph.in_degree(node, weight = 'weight')
                if d != 0:
                    vnodeSE -= (d / self.vol) * math.log2(d / volume)
            self.struc_data[vname] = [volume, cut, vSE, vnodeSE, neighbor_comms]
    
    # 更新结构数据的二维表示
    def update_struc_data_2d(self):
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
                        vmnodeSE = self.struc_data[v1][3] - (self.struc_data[v1][0]/ self.vol) * math.log2(self.struc_data[v1][0] / vm) + \
                            self.struc_data[v2][3] - (self.struc_data[v2][0]/ self.vol) * math.log2(self.struc_data[v2][0] / vm)
                    self.struc_data_2d[k] = [vm, gm, vmSE, vmnodeSE]
    
    # 贪心算法最小化2D SE并更新社区划分
    def update_division_MinSE(self):
    
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
                # change the tree structure: Merge v1 & v2 -> v1
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
                vmnodeSE = self.struc_data[vm1][3] - (self.struc_data[vm1][0]/ self.vol) * math.log2(self.struc_data[vm1][0] / volume) + \
                    self.struc_data[vm2][3] - (self.struc_data[vm2][0]/ self.vol) * math.log2(self.struc_data[vm2][0] / volume)
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
                            vmnodeSE = self.struc_data[v1][3] - (self.struc_data[v1][0]/ self.vol) * math.log2(self.struc_data[v1][0] / vm) + \
                                self.struc_data[v2][3] - (self.struc_data[v2][0]/ self.vol) * math.log2(self.struc_data[v2][0] / vm)
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
                            vmnodeSE = self.struc_data[v1][3] - (self.struc_data[v1][0]/ self.vol) * math.log2(self.struc_data[v1][0] / vm) + \
                                self.struc_data[v2][3] - (self.struc_data[v2][0]/ self.vol) * math.log2(self.struc_data[v2][0] / vm)
                            struc_data_2d_new[k] = [vm, gm, vmSE, vmnodeSE]
                    else:
                        struc_data_2d_new[k] = self.struc_data_2d[k]
                self.struc_data_2d = struc_data_2d_new
            else:
                break
    
    # 初始化社区划分
    def init_division(self):
        self.division = {}
        for node in self.graph.nodes:
            new_comm = node
            self.division[new_comm] = [node]
            self.graph.nodes[node]['comm'] = new_comm

# 贪心的二维结构熵最小化函数
def vanilla_2D_SE_mini(A, division = None):
    '''
    # A = (a_{i2,i1}), where a_{i2,i1} \in {0, 1} for i2, i1 = 1, ..., n. n is the total number of objects, i.e., nodes. 
    # (i2, i1) stands for ((t-1)-th state, t-th state), i.e., (X_{t-1}, X_t).
    # a_{i2,i1} = 1 indicates that there is an edge starts from i2 and points to i1.
    '''
    seg = SE(A)
    SE_1d = seg.calc_1dSE()

    if division is None:
        seg.init_division()
    else:
        seg.division = division

    seg.update_struc_data()
    seg.update_struc_data_2d()
    seg.update_division_MinSE()
    comms = seg.division

    SE_2d = 0
    for vname in seg.division.keys():
        SE_2d += seg.struc_data[vname][2]
        SE_2d += seg.struc_data[vname][3]
    assert math.isclose(SE_2d, seg.calc_2dSE())

    return SE_1d, comms, SE_2d

# 分层的二维结构熵最小化函数
def hier_2D_SE_mini(A, n = 100):
    '''
    # A = (a_{i2,i1}), where a_{i2,i1} \in {0, 1} for i2, i1 = 1, ..., n. n is the total number of objects, i.e., nodes. 
    # (i2, i1) stands for ((t-1)-th state, t-th state), i.e., (X_{t-1}, X_t).
    # a_{i2,i1} = 1 indicates that there is an edge starts from i2 and points to i1.
    '''
    n_clusters = A.shape[0]

    if n >= n_clusters:
        SE_1d, comms, SE_2d = vanilla_2D_SE_mini(A)
        return SE_1d, comms, SE_2d

    seg = SE(A)
    SE_1d = seg.calc_1dSE()

    all_comms = [[i] for i in range(n_clusters)]
    all_sub_comms = [all_comms[i*n: min((i+1)*n, len(all_comms))] for i in range(math.ceil(len(all_comms)/n))]
    while True:
        #print('all_comms', all_comms)
        last_all_comms = copy.deepcopy(all_comms)
        all_comms = []
        for sub_comms in all_sub_comms:
            split = list(chain(*sub_comms))
            #print(' split', split)
            sub_A = A[np.ix_(split, split)]
            sub_seg = SE(sub_A)
            sub_seg.division = split2division(split, sub_comms)
            sub_seg.update_struc_data()
            sub_seg.update_struc_data_2d()
            sub_seg.update_division_MinSE()
            temp = division2split(split, sub_seg.division.values())
            temp.sort()
            all_comms += temp
        if len(all_sub_comms) == 1:
            break
        all_comms.sort()
        if last_all_comms == all_comms:
            n *= 2
        all_sub_comms = [all_comms[i*n: min((i+1)*n, len(all_comms))] for i in range(math.ceil(len(all_comms)/n))]
    seg.division = {i:cluster for i, cluster in enumerate(all_comms)}
    seg.update_struc_data()
    seg.update_struc_data_2d()
    SE_2d = 0
    for vname in seg.division.keys():
        SE_2d += seg.struc_data[vname][2]
        SE_2d += seg.struc_data[vname][3]

    return SE_1d, seg.division, SE_2d

# 分割与划分转换函数
def division2split(split, division_values):
    division2split_map = {d_idx: s_idx for d_idx, s_idx in enumerate(split)}
    return [[division2split_map[d_idx] for d_idx in cluster] for cluster in division_values]

def split2division(split, sub_comms):
    split2division_map = {s_idx: d_idx for d_idx, s_idx in enumerate(split)}
    sub_comms = [[split2division_map[s_idx] for s_idx in cluster] for cluster in sub_comms]
    return {i:cluster for i, cluster in enumerate(sub_comms)}

# 测试函数
def test_SE():
    '''
    Test SE.
    '''
    # This example tensor comes from https://towardsdatascience.com/pagerank-algorithm-fully-explained-dc794184b4af
    # A = (a_{i2,i1}), where a_{i2,i1} \in {0, 1} for i2, i1 = 1, ..., n. n is the total number of objects, i.e., nodes. 
    # (i2, i1) stands for ((t-1)-th state, t-th state), i.e., (X_{t-1}, X_t).
    # a_{i2,i1} = 1 indicates that there is an edge starts from i2 and points to i1.
    # 有向无权图
    A = np.array([
        [0, 0, 1, 1, 1],  # 节点0指向节点2、3、4
        [0, 0, 0, 0, 1],  # 节点1指向节点4
        [0, 1, 0, 1, 0],  # 节点2指向节点1、3
        [0, 1, 0, 0, 0],  # 节点3指向节点1
        [1, 1, 1, 0, 0]   # 节点4指向节点0、1、2
    ]) 

    seg = SE(A)
    print('1D SE: ', seg.calc_1dSE())
    seg.init_division()
    print('Initial 2D SE: ', seg.calc_2dSE())

    SE_1d, comms, SE_2d = vanilla_2D_SE_mini(A)
    print('Minimized 2D SE: ', SE_2d)
    print('Detected communities: ', comms)

    return

if __name__ == "__main__":
    test_SE()


# 《Multi-Relational Structural Entropy》
# 然而，SE 忽略了图关系固有的异质性，SE 假设节点之间只存在单一类型的关系（还不能简单将多关系网络处理成单关系图）。本文扩展SE以考虑异构关系，并提出多关系图结构信息的第一个度量，即多关系结构熵(MrSE)
# 首先通过随机游走的平稳分布的新颖视角来投射SE（即RSSE），通过在每一步同时考虑节点和关系类型的选择扩展到多关系网络

# SE_1.py和SE_2.py
# 相同点
# 结构熵的计算：
    # 两个代码都计算了一维结构熵（1D SE）和二维结构熵（2D SE）。
    # 它们都使用节点的度（有向图中的入度）来计算1D SE。
    # 都通过计算社区的割边总和（cut size）和社区的容量（volume）来计算2D SE。
# 初始化和更新社区划分：
    # 都有初始化社区划分的功能，初始划分为每个节点各自为一个社区。
    # 都通过贪心算法迭代地合并社区，以最小化结构熵。
# 邻接矩阵表示图：
    # 两个代码都接受邻接矩阵作为输入数据，并基于此构建图。
# 使用NetworkX库：
    # 都使用了NetworkX库来处理图数据结构及其相关的操作。

# 不同点
# 图的类型：
    # 第一个代码实现使用无向图（nx.Graph），并基于加权图来计算结构熵。
    # 第二个代码实现使用有向图（nx.DiGraph），并基于二值邻接矩阵来计算结构熵。
# 图的创建：
    # 第一个代码实现通过传入加权边列表来创建图。
    # 第二个代码实现通过传入邻接矩阵来创建图。
# 计算割边总和和社区容量的方法：
    # 第一个代码实现直接使用NetworkX的cuts.cut_size和cuts.volume函数。
    # 第二个代码实现通过自定义方法来计算割边总和和社区容量。
# 邻居社区的计算：
    # 第一个代码实现没有特别处理邻居社区的计算。
    # 第二个代码实现通过遍历邻接矩阵来确定每个社区的邻居社区。
# 结构数据的存储：
    # 第一个代码实现使用self.struc_data和self.struc_data_2d来存储社区的结构数据及其合并后的数据。
    # 第二个代码实现也使用了类似的结构来存储社区的数据，但存储内容和计算方法稍有不同。
# 贪心算法的细节：
    # 第一个代码实现中的贪心算法直接通过计算合并社区后的熵变化来确定是否合并社区。
    # 第二个代码实现中的贪心算法也通过计算合并后的熵变化来确定合并，但有更详细的邻居社区更新过程。