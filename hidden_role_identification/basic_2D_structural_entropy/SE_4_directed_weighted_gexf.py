import networkx as nx
import math
from networkx.algorithms import cuts
from itertools import chain
import copy

class SE:
    def __init__(self, graph):
        '''
        # 使用GraphML文件中的图结构进行初始化。
        # graph是从GraphML文件读取的networkx.DiGraph对象。
        '''
        self.graph = graph
        self.vol = self.get_g_vol()
        self.division = {}  # {comm1: [node11, node12, ...], comm2: [node21, node22, ...], ...}
        self.struc_data = {}  # {comm1: [vol1, cut1, community_node_SE, leaf_nodes_SE, neighbor_comms1], comm2:[vol2, cut2, community_node_SE, leaf_nodes_SE, neighbor_comms2]，... }
        self.struc_data_2d = {} # {(comm1, comm2): [vol_after_merge, cut_after_merge, comm_node_SE_after_merge, leaf_nodes_SE_after_merge], (comm1, comm3): [], ...}

    def get_g_vol(self):
        return cuts.volume(self.graph, self.graph.nodes, weight='transfer_value')

    def get_cut(self, comm):
        comm_set = {n for n in comm if n in self.graph}
        all_in_edges = self.graph.in_edges(nbunch=comm_set, data=True)  # 获取所有边及其属性
        cut_in_edges = (e for e in all_in_edges if e[0] not in comm_set)
        
        # 从边属性中提取 transfer_value 作为权重，如果不存在则为0
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
            
            # 检查 v 和 self.vol 的值，防止对数错误
            if v > 0 and self.vol > 0:
                SE += - (g / self.vol) * math.log2(v / self.vol)
            
            for node in comm:
                d = self.graph.in_degree(node, weight='transfer_value')
                # 检查 d 和 v 的值，防止对数错误
                if d > 0 and v > 0:
                    SE += - (d / self.vol) * math.log2(d / v)
        return SE


    def init_division(self):
        """初始化每个节点为一个独立的社区"""
        self.division = {}
        for node in self.graph.nodes:
            new_comm = node
            self.division[new_comm] = [node]
            self.graph.nodes[node]['comm'] = new_comm

    def update_struc_data(self):
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

def hier_2D_SE_mini(graph, n=100):
    '''
    分层的2D结构熵最小化方法。
    graph 是从GraphML读取的图，n 表示分块的大小
    '''
    n_clusters = graph.number_of_nodes()

    if n >= n_clusters:
        SE_1d, comms, SE_2d = vanilla_2D_SE_mini(graph)
        return SE_1d, comms, SE_2d

    seg = SE(graph)
    SE_1d = seg.calc_1dSE()

    all_comms = [[i] for i in range(n_clusters)]
    all_sub_comms = [all_comms[i*n: min((i+1)*n, len(all_comms))] for i in range(math.ceil(len(all_comms)/n))]
    
    while True:
        last_all_comms = copy.deepcopy(all_comms)
        all_comms = []
        for sub_comms in all_sub_comms:
            split = list(chain(*sub_comms))
            sub_graph = graph.subgraph(split)
            sub_seg = SE(sub_graph)
            sub_seg.init_division()
            sub_seg.update_struc_data()
            sub_seg.update_struc_data_2d()
            sub_seg.update_division_MinSE()
            temp = list(sub_seg.division.values())
            temp.sort()
            all_comms += temp

        if len(all_sub_comms) == 1:
            break
        all_comms.sort()
        if last_all_comms == all_comms:
            n *= 2
        all_sub_comms = [all_comms[i*n: min((i+1)*n, len(all_comms))] for i in range(math.ceil(len(all_comms)/n))]

    seg.division = {i: cluster for i, cluster in enumerate(all_comms)}
    seg.update_struc_data()
    seg.update_struc_data_2d()
    
    SE_2d = 0
    for vname in seg.division.keys():
        SE_2d += seg.struc_data[vname][2]
        SE_2d += seg.struc_data[vname][3]

    return SE_1d, seg.division, SE_2d

def vanilla_2D_SE_mini(graph, division=None):
    '''
    # 使用从GraphML文件读取的graph对象来进行2D结构熵的最小化。
    '''
    seg = SE(graph)
    SE_1d = seg.calc_1dSE()

    if division is None:
        seg.init_division()  # 使用init_division初始化分区
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

def test_SE():
    '''
    测试SE类的1D和2D结构熵计算，包括分层结构熵最小化。
    '''
    # 读取GraphML文件
    graph = nx.read_graphml("/home/lab0/gnn-wjx/0xc2a81eb482cb4677136d8812cc6db6e0cb580883.graphml")

    # 测试 vanilla_2D_SE_mini 函数
    print("Testing vanilla 2D SE minimization...")
    SE_1d, comms, SE_2d = vanilla_2D_SE_mini(graph)
    print(f"1D SE: {SE_1d}")
    print(f"Minimized 2D SE: {SE_2d}")
    print(f"Detected communities: {comms}")
    print(f"Number of detected communities: {len(comms)}")  # 输出社区个数

    # 测试 hier_2D_SE_mini 函数
    # print("\nTesting hierarchical 2D SE minimization...")
    # SE_1d, comms, SE_2d = hier_2D_SE_mini(graph, n=2)
    # print(f"1D SE: {SE_1d}")
    # print(f"Hierarchical minimized 2D SE: {SE_2d}")
    # print(f"Detected communities: {comms}")
    # print(f"Number of detected communities: {len(comms)}")  # 输出社区个数



if __name__ == "__main__":
    test_SE()