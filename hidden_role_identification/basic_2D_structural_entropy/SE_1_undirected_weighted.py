# ·  为什么要用结构熵？
#      ·  角色是结构相似的一组节点集合
#      ·  现有无监督图结构学习缺乏鲁棒性【结果不稳定，需要减轻图扰动的干扰并在没有标签情况下学习节点适当表示】和可解释性【分成社区】
#      ·  现有聚类问题中，需要提前预定义角色结构和数量，且传统的欧几里得距离无法充分捕捉图的复杂结构和节点之间的非线性关系
#      ·  结构熵无法直接应用加权网络，经典定义缺乏建模节点特征
# ·  怎么加入结构熵？
#      ·  熵可以新命名？
#      ·  1D加入到选举投票算法里
#      ·  邻域局部结构特征（Attri-GAT）+全局结构特征【结构熵】
#      ·  将之前的自监督学习任务改成最小化结构熵来实现
#      ·  同时可以得到角色自适应聚类【编码树表示将图多粒度划分为分层社区和子社区，从而提供了更好的可解释性途径】

# 加权无向图的一维、二维结构熵最小化
import networkx as nx
from networkx.algorithms import cuts
import math
from itertools import chain

# 类SE的定义
class SE:
    def __init__(self, graph: nx.Graph):
        self.graph = graph.copy()
        self.vol = self.get_vol()
        self.division = {}  # {comm1: [node11, node12, ...], comm2: [node21, node22, ...], ...}
        self.struc_data = {}  # {comm1: [vol1, cut1, community_node_SE, leaf_nodes_SE], comm2:[vol2, cut2, community_node_SE, leaf_nodes_SE]，... }
        self.struc_data_2d = {} # {comm1: {comm2: [vol_after_merge, cut_after_merge, comm_node_SE_after_merge, leaf_nodes_SE_after_merge], comm3: [], ...}, ...}
    
    # 计算图的总容量（volume）
    def get_vol(self):
        '''
        get the volume of the graph
        '''
        return cuts.volume(self.graph, self.graph.nodes, weight = 'weight')
    
    # 计算图的一维结构熵（1D SE）
    def calc_1dSE(self):
        '''
        get the 1D SE of the graph
        '''
        SE = 0
        for n in self.graph.nodes:
            d = cuts.volume(self.graph, [n], weight = 'weight')
            SE += - (d / self.vol) * math.log2(d / self.vol)
        return SE
    
    # 更新图的一维结构熵（1D SE）
    def update_1dSE(self, original_1dSE, new_edges):
        '''
        get the updated 1D SE after new edges are inserted into the graph
        '''
    
        affected_nodes = []
        for edge in new_edges:
            affected_nodes += [edge[0], edge[1]]
        affected_nodes = set(affected_nodes)

        original_vol = self.vol
        original_degree_dict = {node:0 for node in affected_nodes}
        for node in affected_nodes.intersection(set(self.graph.nodes)):
            original_degree_dict[node] = self.graph.degree(node, weight = 'weight')

        # insert new edges into the graph
        self.graph.add_weighted_edges_from(new_edges)

        self.vol = self.get_vol()
        updated_vol = self.vol
        updated_degree_dict = {}
        for node in affected_nodes:
            updated_degree_dict[node] = self.graph.degree(node, weight = 'weight')
        
        updated_1dSE = (original_vol / updated_vol) * (original_1dSE - math.log2(original_vol / updated_vol))
        for node in affected_nodes:
            d_original = original_degree_dict[node]
            d_updated = updated_degree_dict[node]
            if d_original != d_updated:
                if d_original != 0:
                    updated_1dSE += (d_original / updated_vol) * math.log2(d_original / updated_vol)
                updated_1dSE -= (d_updated / updated_vol) * math.log2(d_updated / updated_vol)

        return updated_1dSE
    
    # 计算社区的割边总和（cut size）和容量（volume）
    def get_cut(self, comm):
        '''
        get the sum of the degrees of the cut edges of community comm
        '''
        return cuts.cut_size(self.graph, comm, weight = 'weight')

    def get_volume(self, comm):
        '''
        get the volume of community comm
        '''
        return cuts.volume(self.graph, comm, weight = 'weight')
    
    # 计算图的二维结构熵（2D SE）
    def calc_2dSE(self):
        '''
        get the 2D SE of the graph
        '''
        SE = 0
        for comm in self.division.values():
            g = self.get_cut(comm)
            v = self.get_volume(comm)
            SE += - (g / self.vol) * math.log2(v / self.vol)
            for node in comm:
                d = self.graph.degree(node, weight = 'weight')
                SE += - (d / self.vol) * math.log2(d / v)
        return SE
    
    def show_division(self):
        print(self.division)

    def show_struc_data(self):
        print(self.struc_data)
    
    def show_struc_data_2d(self):
        print(self.struc_data_2d)
        
    def print_graph(self):
        fig, ax = plt.subplots()
        nx.draw(self.graph, ax=ax, with_labels=True)
        plt.show()
    
    # 更新结构数据
    def update_struc_data(self):
        '''
        calculate the volume, cut, communitiy mode SE, and leaf nodes SE of each cummunity, 
        then store them into self.struc_data
        '''
        self.struc_data = {} # {comm1: [vol1, cut1, community_node_SE, leaf_nodes_SE], comm2:[vol2, cut2, community_node_SE, leaf_nodes_SE]，... }
        for vname in self.division.keys():
            comm = self.division[vname]
            volume = self.get_volume(comm)
            cut = self.get_cut(comm)
            if volume == 0:
                vSE = 0
            else:
                vSE = - (cut / self.vol) * math.log2(volume / self.vol)
            vnodeSE = 0
            for node in comm:
                d = self.graph.degree(node, weight = 'weight')
                if d != 0:
                    vnodeSE -= (d / self.vol) * math.log2(d / volume)
            self.struc_data[vname] = [volume, cut, vSE, vnodeSE]
    
    # # 更新结构数据的二维表示
    def update_struc_data_2d(self):
        '''
        calculate the volume, cut, communitiy mode SE, and leaf nodes SE after merging each pair of cummunities, 
        then store them into self.struc_data_2d
        '''
        self.struc_data_2d = {} # {(comm1, comm2): [vol_after_merge, cut_after_merge, comm_node_SE_after_merge, leaf_nodes_SE_after_merge], (comm1, comm3): [], ...}
        comm_num = len(self.division)
        for i in range(comm_num):
            for j in range(i + 1, comm_num):
                v1 = list(self.division.keys())[i]
                v2 = list(self.division.keys())[j]
                if v1 < v2:
                    k = (v1, v2)
                else:
                    k = (v2, v1)

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

    # 初始化社区划分
    def init_division(self):
        '''
        initialize self.division such that each node assigned to its own community
        '''
        self.division = {}
        for node in self.graph.nodes:
            new_comm = node
            self.division[new_comm] = [node]
            self.graph.nodes[node]['comm'] = new_comm

    # 增加孤立节点到图中
    def add_isolates(self):
        '''
        add any isolated nodes into graph
        '''
        all_nodes = list(chain(*list(self.division.values())))
        all_nodes.sort()
        edge_nodes = list(self.graph.nodes)
        edge_nodes.sort()
        if all_nodes != edge_nodes:
            for node in set(all_nodes)-set(edge_nodes):
                self.graph.add_node(node)
    
    # 更新社区划分以最小化2D SE
    def update_division_MinSE(self):
        '''
        greedily update the encoding tree to minimize 2D SE
        '''
        def Mg_operator(v1, v2):
            '''
            MERGE operator. It calculates the delta SE caused by mergeing communities v1 and v2, 
            without actually merging them, i.e., the encoding tree won't be changed
            '''
            v1SE = self.struc_data[v1][2] 
            v1nodeSE = self.struc_data[v1][3]

            v2SE = self.struc_data[v2][2]
            v2nodeSE = self.struc_data[v2][3]

            if v1 < v2:
                k = (v1, v2)
            else:
                k = (v2, v1)
            vm, gm, vmSE, vmnodeSE = self.struc_data_2d[k]
            delta_SE = vmSE + vmnodeSE - (v1SE + v1nodeSE + v2SE + v2nodeSE)
            return delta_SE

        # continue merging any two communities that can cause the largest decrease in SE, 
        # until the SE can't be further reduced
        while True: 
            comm_num = len(self.division)
            delta_SE = 99999
            vm1 = None
            vm2 = None
            for i in range(comm_num):
                for j in range(i + 1, comm_num):
                    v1 = list(self.division.keys())[i]
                    v2 = list(self.division.keys())[j]
                    new_delta_SE = Mg_operator(v1, v2)
                    if new_delta_SE < delta_SE:
                        delta_SE = new_delta_SE
                        vm1 = v1
                        vm2 = v2

            if delta_SE < 0:
                # Merge v2 into v1, and update the encoding tree accordingly
                for node in self.division[vm2]:
                    self.graph.nodes[node]['comm'] = vm1
                self.division[vm1] += self.division[vm2]
                self.division.pop(vm2)

                volume = self.struc_data[vm1][0] + self.struc_data[vm2][0]
                cut = self.get_cut(self.division[vm1])
                vmSE = - (cut / self.vol) * math.log2(volume / self.vol)
                vmnodeSE = self.struc_data[vm1][3] - (self.struc_data[vm1][0]/ self.vol) * math.log2(self.struc_data[vm1][0] / volume) + \
                        self.struc_data[vm2][3] - (self.struc_data[vm2][0]/ self.vol) * math.log2(self.struc_data[vm2][0] / volume)
                self.struc_data[vm1] = [volume, cut, vmSE, vmnodeSE]
                self.struc_data.pop(vm2)

                struc_data_2d_new = {}
                for k in self.struc_data_2d.keys():
                    if k[0] == vm2 or k[1] == vm2:
                        continue
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

# 贪心的二维结构熵最小化函数
def vanilla_2D_SE_mini(weighted_edges):
    '''
    vanilla (greedy) 2D SE minimization
    '''
    g = nx.Graph()
    g.add_weighted_edges_from(weighted_edges)
    
    seg = SE(g)
    seg.init_division()
    #seg.show_division()
    SE1D = seg.calc_1dSE()

    seg.update_struc_data()
    #seg.show_struc_data()
    seg.update_struc_data_2d()
    #seg.show_struc_data_2d()
    initial_SE2D = seg.calc_2dSE()

    seg.update_division_MinSE()
    communities = seg.division
    minimized_SE2D = seg.calc_2dSE()

    return SE1D, initial_SE2D, minimized_SE2D, communities


# 测试函数
def test_vanilla_2D_SE_mini():
    # 小型加权无向图来进行演示。具体来说，测试用的网络包含以下两条边：节点1和节点2之间的边，权重为2，节点1和节点3之间的边，权重为4
    weighted_edges = [(1, 2, 2), (1, 3, 4)]

    g = nx.Graph()
    g.add_weighted_edges_from(weighted_edges)
    A = nx.adjacency_matrix(g).todense()
    print('adjacency matrix: \n', A)
    print('g.nodes: ', g.nodes)
    print('g.edges: ', g.edges)
    print('degrees of nodes: ', list(g.degree(g.nodes, weight = 'weight')))

    SE1D, initial_SE2D, minimized_SE2D, communities = vanilla_2D_SE_mini(weighted_edges)
    print('\n1D SE of the graph: ', SE1D)
    print('initial 2D SE of the graph: ', initial_SE2D)
    print('the minimum 2D SE of the graph: ', minimized_SE2D)
    print('communities detected: ', communities)
    return

if __name__ == "__main__":
    test_vanilla_2D_SE_mini()

# Hierarchical and Incremental Structural Entropy Minimization for Unsupervised Social Event Detection
# 社交事件检测作为舆情挖掘、假新闻检测等的基础，越来越受到工业界和学术界的关注。现有研究通常将社交事件检测任务形式化为从社交媒体序列中提取相关消息簇消息。近年来，基于图神经网络的社会事件检测研究蓬勃发展，然而现有GNN方法在构建消息图时，通常仅仅考虑共享属性的消息之间的连接，忽略了语义相近但没有共同属性的消息之间的相关性【挑战1】。此外，这些模型的 GNN 组件需要监督训练并预先确定用于预测的事件总数【挑战2】。 最近的基于 GNN 的方法开始采用对比学习、归纳学习和伪标签生成来缓解对标签的依赖。 然而，对于初始培训和定期维护，手动标记仍然是必要的。
# 本文通过图结构熵最小化来解决社交事件检测。在保留基于 GNN 的方法的优点的同时，所提出的框架 HISEvent 构造了信息更丰富的消息图，是无监督的，并且不需要先验给定事件的数量。
