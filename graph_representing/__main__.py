import os
import os.path as osp
import networkx
import pickle
import json
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric.transforms as T
from torch_geometric.nn import (
    GINEConv,
    GPSConv,
    global_add_pool,
    AntiSymmetricConv,
    SuperGATConv,
    FAConv,
    GeneralConv,
    GCNConv,
    EGConv,
    GATv2Conv,
    FiLMConv,
    PNAConv,
)
from torch_geometric.nn.conv.dir_gnn_conv import DirGNNConv

import torch
import torch.nn.functional as F
from torch_geometric.nn import VGAE, GAE
from sklearn.cluster import KMeans
from torch_geometric.utils import from_networkx
import torch
from sklearn.cluster import KMeans

import torch_geometric.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PNAEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PNAEncoder, self).__init__()
        self.conv1 = PNAConv(in_channels, 2 * out_channels)
        self.conv2 = PNAConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x.float(), edge_index.long()).relu()
        return self.conv2(x, edge_index.long())


class GATv2Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATv2Encoder, self).__init__()
        self.conv1 = GATv2Conv(in_channels, 2 * out_channels)
        self.conv2 = GATv2Conv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x.float(), edge_index.long()).relu()
        return self.conv2(x, edge_index.long())


class FiLMEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FiLMEncoder, self).__init__()
        self.conv1 = FiLMConv(in_channels, 2 * out_channels)
        self.conv2 = FiLMConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x.float(), edge_index.long()).relu()
        return self.conv2(x, edge_index.long())


class DirGNNWithGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DirGNNWithGCNEncoder, self).__init__()
        self.conv1 = DirGNNConv(GCNConv(in_channels, 2 * out_channels))
        self.conv2 = DirGNNConv(GCNConv(2 * out_channels, out_channels))

    def forward(self, x, edge_index):
        x = self.conv1(x.float(), edge_index.long()).relu()
        return self.conv2(x, edge_index.long())


class SuperGATEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SuperGATEncoder, self).__init__()
        self.conv1 = SuperGATConv(in_channels, 2 * out_channels)
        self.conv2 = SuperGATConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x.float(), edge_index.long()).relu()
        return self.conv2(x, edge_index.long())


class EGCEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EGCEncoder, self).__init__()
        self.conv1 = EGConv(in_channels, 2 * out_channels)
        self.conv2 = EGConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x.float(), edge_index.long()).relu()
        return self.conv2(x, edge_index.long())


encoder = FiLMEncoder(train_data.num_features, out_channels=32)
model = GAE(encoder).to(device)

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.005)


def train():
    model.train()
    encoder_optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index.int())

    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    loss.backward()
    encoder_optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)

    auc, ap = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

    return auc, ap


for epoch in range(1, 151):
    loss = train()
    auc, ap = test(test_data)
    print((f"Epoch: {epoch:03d}, Loss: {loss:.3f}, AUC: {auc:.3f}, " f"AP: {ap:.3f},"))


@torch.no_grad()
def get_result(data):
    model.eval()
    z = model.encode(data.x, data.edge_index.int())
    # Cluster embedded values using k-means.
    kmeans_input = z.cpu().numpy()
    kmeans = KMeans(n_clusters=7).fit(kmeans_input)
    pred = kmeans.predict(kmeans_input)

    result = {}
    for i in range(len(pred)):
        # addr = mapping_reversed[i]
        # result[addr] = int(pred[i])
        result[i] = int(pred[i])

    return result



if __name__ == "__main__":
    # load one by one
    for filename in os.listdir("dataset/pos_nx_normalized"):
    # filename = "0xc2a81eb482cb4677136d8812cc6db6e0cb580883.pkl"
        nxG = pickle.load(open(f"./dataset/pos_nx_normalized/{filename}", "rb"))

        mapping = {}
        mapping_reversed = {}
        for id, data in nxG.nodes(data=True):
            # print(data)
            addr = data["addr"]
            mapping[addr] = id
            mapping_reversed[id] = addr
        node_attrs = [
            "STOP",
            "ADD",
            "MUL",
            "SUB",
            "DIV",
            "SDIV",
            "MOD",
            "SMOD",
            "ADDMOD",
            "MULMOD",
            "EXP",
            "SIGNEXTEND",
            "LT",
            "GT",
            "SLT",
            "SGT",
            "EQ",
            "ISZERO",
            "AND",
            "OR",
            "XOR",
            "NOT",
            "BYTE",
            "SHL",
            "SHR",
            "SAR",
            "SHA3",
            "ADDRESS",
            "BALANCE",
            "ORIGIN",
            "CALLER",
            "CALLVALUE",
            "CALLDATALOAD",
            "CALLDATASIZE",
            "CALLDATACOPY",
            "CODESIZE",
            "CODECOPY",
            "GASPRICE",
            "EXTCODESIZE",
            "EXTCODECOPY",
            "RETURNDATASIZE",
            "RETURNDATACOPY",
            "EXTCODEHASH",
            "BLOCKHASH",
            "COINBASE",
            "TIMESTAMP",
            "NUMBER",
            "DIFFICULTY",
            "GASLIMIT",
            "CHAINID",
            "SELFBALANCE",
            "POP",
            "MLOAD",
            "MSTORE",
            "MSTORE8",
            "SLOAD",
            "SSTORE",
            "JUMP",
            "JUMPI",
            "GETPC",
            "MSIZE",
            "GAS",
            "JUMPDEST",
            "PUSH1",
            "PUSH2",
            "PUSH3",
            "PUSH4",
            "PUSH5",
            "PUSH6",
            "PUSH7",
            "PUSH8",
            "PUSH9",
            "PUSH10",
            "PUSH11",
            "PUSH12",
            "PUSH13",
            "PUSH14",
            "PUSH15",
            "PUSH16",
            "PUSH17",
            "PUSH18",
            "PUSH19",
            "PUSH20",
            "PUSH21",
            "PUSH22",
            "PUSH23",
            "PUSH24",
            "PUSH25",
            "PUSH26",
            "PUSH27",
            "PUSH28",
            "PUSH29",
            "PUSH30",
            "PUSH31",
            "PUSH32",
            "DUP1",
            "DUP2",
            "DUP3",
            "DUP4",
            "DUP5",
            "DUP6",
            "DUP7",
            "DUP8",
            "DUP9",
            "DUP10",
            "DUP11",
            "DUP12",
            "DUP13",
            "DUP14",
            "DUP15",
            "DUP16",
            "SWAP1",
            "SWAP2",
            "SWAP3",
            "SWAP4",
            "SWAP5",
            "SWAP6",
            "SWAP7",
            "SWAP8",
            "SWAP9",
            "SWAP10",
            "SWAP11",
            "SWAP12",
            "SWAP13",
            "SWAP14",
            "SWAP15",
            "SWAP16",
            "LOG0",
            "LOG1",
            "LOG2",
            "LOG3",
            "LOG4",
            "CREATE",
            "CALL",
            "CALLCODE",
            "RETURN",
            "DELEGATECALL",
            "CREATE2",
            "STATICCALL",
            "REVERT",
            "INVALID",
            "SELFDESTRUCT",
            "y",
            "sum_val_in",
            "sum_val_out",
            "avg_val_in",
            "avg_val_out",
            "count_in",
            "count_out",
            "count",
            "freq",
            "freq_in",
            "freq_out",
            "gini_val",
            "gini_val_in",
            "gini_val_out",
            "avg_gas",
            "avg_gas_in",
            "avg_gas_out",
            "avg_gasprice",
            "avg_gasprice_in",
            "avg_gasprice_out",
            "in_out_rate",
            "addr",
        ]
        node_attrs.remove("addr")
        node_attrs.remove("y")
        # networkx.relabel_nodes(nxG, mapping)
        data = from_networkx(
            nxG,
            group_edge_attrs=[
                "block_number",
                "transfer_value",
                "gas_used",
                "effective_gas_price",
            ],
            group_node_attrs=node_attrs,
        )
        data.x = torch.nan_to_num(data.x, nan=0.0)
        transform = T.Compose(
            [
                T.ToDevice(device),
                T.NormalizeFeatures(),
                T.RandomLinkSplit(
                    num_val=0.00,
                    num_test=0.1,
                    is_undirected=True,
                    split_labels=True,
                    add_negative_train_samples=False,
                ),
            ]
        )
        train_data, val_data, test_data = transform(data)

        result = get_result(data)
        with open(f"graph_representing_result/{filename}.json", "w") as f:
            json.dump(result, f)
            