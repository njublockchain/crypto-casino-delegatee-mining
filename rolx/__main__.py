# %%
# set/export OPENBLAS_NUM_THREADS=1
from graphrole import RecursiveFeatureExtractor, RoleExtractor
import pickle, json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
import warnings
import networkx as nx

# choices = [
#     '0x419d0d8bdd9af5e606ae2232ed285aff190e711b',
#     '0x4740735aa98dc8aa232bd049f8f0210458e7fca3',
#     '0xc237868a9c5729bdf3173dddacaa336a0a5bb6e0',
#     '0xe477292f1b3268687a29376116b0ed27a9c76170',
#     '0x5eeaa2dcb23056f4e8654a349e57ebe5e76b5e6e',
#     '0x9b68bfae21df5a510931a262cecf63f41338f264',
#     '0x368c5290b13caa10284db58b4ad4f3e9ee8bf4c9',
#     '0xc2a81eb482cb4677136d8812cc6db6e0cb580883',
#     '0xa3ad8c7ab6b731045b5b16e3fdf77975c71abe79',
#     '0x2d886570a0da04885bfd6eb48ed8b8ff01a0eb7e',
#     '0x73b708e84837ffccde2933e3a1531fe61d5e80ef',
#     # '0x2c644c3bbea053ed95a6bc04a94c9ce928ff9881',
#     '0xe9f1d62c671efe99896492766c0b416bd3fb9e52',
#     '0x9d7107c8e30617cadc11f9692a19c82ae8bba938']


# %%
# node_attributes = set()
# for _, data in nxG.nodes(data=True):
#     node_attributes.update(data.keys())

# ['ADD',
#  'ADDMOD',
#  'ADDRESS',
#  'AND',
#  'BALANCE',
#  'BLOCKHASH',
#  'BYTE',
#  'CALL',
#  'CALLCODE',
#  'CALLDATACOPY',
#  'CALLDATALOAD',
#  'CALLDATASIZE',
#  'CALLER',
#  'CALLVALUE',
#  'CHAINID',
#  'CODECOPY',
#  'CODESIZE',
#  'COINBASE',
#  'CREATE',
#  'CREATE2',
#  'DELEGATECALL',
#  'DIFFICULTY',
#  'DIV',
#  'DUP1',
#  'DUP10',
#  'DUP11',
#  'DUP12',
#  'DUP13',
#  'DUP14',
#  'DUP15',
#  'DUP16',
#  'DUP2',
#  'DUP3',
#  'DUP4',
#  'DUP5',
#  'DUP6',
#  'DUP7',
#  'DUP8',
#  'DUP9',
#  'EQ',
#  'EXP',
#  'EXTCODECOPY',
#  'EXTCODEHASH',
#  'EXTCODESIZE',
#  'GAS',
#  'GASLIMIT',
#  'GASPRICE',
#  'GETPC',
#  'GT',
#  'INVALID',
#  'ISZERO',
#  'JUMP',
#  'JUMPDEST',
#  'JUMPI',
#  'LOG0',
#  'LOG1',
#  'LOG2',
#  'LOG3',
#  'LOG4',
#  'LT',
#  'MLOAD',
#  'MOD',
#  'MSIZE',
#  'MSTORE',
#  'MSTORE8',
#  'MUL',
#  'MULMOD',
#  'NOT',
#  'NUMBER',
#  'OR',
#  'ORIGIN',
#  'POP',
#  'PUSH1',
#  'PUSH10',
#  'PUSH11',
#  'PUSH12',
#  'PUSH13',
#  'PUSH14',
#  'PUSH15',
#  'PUSH16',
#  'PUSH17',
#  'PUSH18',
#  'PUSH19',
#  'PUSH2',
#  'PUSH20',
#  'PUSH21',
#  'PUSH22',
#  'PUSH23',
#  'PUSH24',
#  'PUSH25',
#  'PUSH26',
#  'PUSH27',
#  'PUSH28',
#  'PUSH29',
#  'PUSH3',
#  'PUSH30',
#  'PUSH31',
#  'PUSH32',
#  'PUSH4',
#  'PUSH5',
#  'PUSH6',
#  'PUSH7',
#  'PUSH8',
#  'PUSH9',
#  'RETURN',
#  'RETURNDATACOPY',
#  'RETURNDATASIZE',
#  'REVERT',
#  'SAR',
#  'SDIV',
#  'SELFBALANCE',
#  'SELFDESTRUCT',
#  'SGT',
#  'SHA3',
#  'SHL',
#  'SHR',
#  'SIGNEXTEND',
#  'SLOAD',
#  'SLT',
#  'SMOD',
#  'SSTORE',
#  'STATICCALL',
#  'STOP',
#  'SUB',
#  'SWAP1',
#  'SWAP10',
#  'SWAP11',
#  'SWAP12',
#  'SWAP13',
#  'SWAP14',
#  'SWAP15',
#  'SWAP16',
#  'SWAP2',
#  'SWAP3',
#  'SWAP4',
#  'SWAP5',
#  'SWAP6',
#  'SWAP7',
#  'SWAP8',
#  'SWAP9',
#  'TIMESTAMP',
#  'XOR',
#  'avg_gas',
#  'avg_gas_in',
#  'avg_gas_out',
#  'avg_gasprice',
#  'avg_gasprice_in',
#  'avg_gasprice_out',
#  'avg_val_in',
#  'avg_val_out',
#  'count',
#  'count_in',
#  'count_out',
#  'freq',
#  'freq_in',
#  'freq_out',
#  'gini_val',
#  'gini_val_in',
#  'gini_val_out',
#  'in_out_rate',
#  'sum_val_in',
#  'sum_val_out',
#  'y']
# ["STOP","ADD","MUL","SUB","DIV","EXP","SIGNEXTEND","LT","GT","SLT","EQ","ISZERO","AND","OR","NOT","SHL","SHR","SHA3","ADDRESS","CALLER","CALLVALUE","CALLDATALOAD","CALLDATASIZE","EXTCODESIZE","RETURNDATASIZE","RETURNDATACOPY","POP","MLOAD","MSTORE","SLOAD","SSTORE","JUMP","JUMPI","GAS","JUMPDEST","PUSH1","PUSH2","PUSH3","PUSH4","PUSH5","PUSH8","PUSH14","PUSH20","PUSH32","DUP1","DUP2","DUP3","DUP4","DUP5","DUP6","DUP7","DUP8","DUP9","DUP10","DUP11","DUP12","DUP13","DUP14","SWAP1","SWAP2","SWAP3","SWAP4","SWAP5","SWAP6","SWAP7","SWAP8","CALL","RETURN","STATICCALL","REVERT","INVALID"]

# %%

# def normalize_df(df: pd.DataFrame):
#     for col in df.columns:
#         if str(col) ==  "attribute_freq":
#             continue
#         ds = df[col]
#         print(ds)
#         df.loc[:][col] = (ds - ds.min()) / (ds.max() - ds.min())
#         return df

# X = normalize_df(features)

# %%

from multiprocessing import Pool


def extract_features(choice):
    nxG = pickle.load(
        open(
            f"/home/ta/3rd-Macao-online-casino/pos_nx0/{choice}.pkl",
            "rb",
        )
    )
    attrs_to_del = [
        "MSTORE8",
        "SWAP16",
        "GASPRICE",
        "PUSH12",
        "XOR",
        "DUP16",
        "LOG1",
        "PUSH11",
        "BYTE",
        "TIMESTAMP",
        "SAR",
        "DUP15",
        "SWAP9",
        "SGT",
        "SDIV",
        "SELFBALANCE",
        "PUSH23",
        "SELFDESTRUCT",
        "PUSH10",
        "SMOD",
        "PUSH25",
        "MSIZE",
        "PUSH6",
        "CODECOPY",
        "COINBASE",
        "PUSH15",
        "PUSH16",
        "PUSH31",
        "PUSH19",
        "LOG2",
        "PUSH30",
        "PUSH21",
        "PUSH22",
        "ADDMOD",
        "PUSH27",
        "SWAP10",
        "PUSH7",
        "CALLDATACOPY",
        "CREATE2",
        "PUSH13",
        "SWAP15",
        "SWAP12",
        "PUSH24",
        "SWAP11",
        "CHAINID",
        "ORIGIN",
        "SWAP14",
        "LOG3",
        "PUSH28",
        "EXTCODECOPY",
        "CALLCODE",
        "MULMOD",
        "SWAP13",
        "GASLIMIT",
        "DELEGATECALL",
        "MOD",
        "EXTCODEHASH",
        "PUSH9",
        "GETPC",
        "PUSH26",
        "BALANCE",
        "CODESIZE",
        "NUMBER",
        "CREATE",
        "PUSH17",
        "PUSH29",
        "PUSH18",
        "BLOCKHASH",
        "LOG4",
        "LOG0",
        "DIFFICULTY",
    ]

    nxG = nx.convert_node_labels_to_integers(nxG, label_attribute="addr")

    gini_vals = pd.Series(nx.get_node_attributes(nxG, "gini_val"))
    gini_val_ins = pd.Series(nx.get_node_attributes(nxG, "gini_val_in"))
    gini_val_outs = pd.Series(nx.get_node_attributes(nxG, "gini_val_out"))

    def normalize_ds(ds):
        return (ds - ds.min()) / (ds.max() - ds.min())

    nx.set_node_attributes(nxG, normalize_ds(gini_vals), "gini_val")
    nx.set_node_attributes(nxG, normalize_ds(gini_val_ins), "gini_val_in")
    nx.set_node_attributes(nxG, normalize_ds(gini_val_outs), "gini_val_out")

    with open(f"pos_nx_normalized/{choice}.pkl", "wb") as f:
        pickle.dump(nxG, f)

    for node in nxG.nodes:
        for attr_name in attrs_to_del:
            if attr_name in nxG.nodes[node]:
                del nxG.nodes[node][attr_name]
                continue
            if np.isnan(nxG.nodes[node][attr_name]):
                nxG.nodes[node][attr_name] = 0.0

    feature_extractor = RecursiveFeatureExtractor(nxG, attributes=True)
    features = feature_extractor.extract_features()
    features.to_csv(f"roles_X/{choice}.gX.csv")

    return features


def f(choice):
    if not os.path.exists(f"roles_X/{choice}.gX.csv"):
        print(f"cannot find features for {choice}, start extracting...")
        features = extract_features(choice)
        print(features)
    else:
        features = pd.read_csv(f"roles_X/{choice}.gX.csv")

    features.fillna(0)

    role_extractor = RoleExtractor(n_roles=None)
    # role_extractor = RoleExtractor(n_roles=7)
    # try:
    role_extractor.extract_role_factors(features)
    # except Exception as e:
    #     print(f"failed to extract roles for {choice}:", e)
    #     return

    with open(f"roles/{choice}.pkl", "wb") as f:
        pickle.dump(role_extractor, f)

    node_roles = role_extractor.roles
    print("\nNode role assignments:")
    pprint(node_roles)
    print("\nNode role membership by percentage:")
    print(role_extractor.role_percentage)

    # # build color palette for plotting
    # unique_roles = sorted(set(node_roles.values()))
    # color_map = sns.color_palette('Paired', n_colors=len(unique_roles))
    # # map roles to colors
    # role_colors = {role: color_map[i] for i, role in enumerate(unique_roles)}
    # # build list of colors for all nodes in G
    # node_colors = [role_colors[node_roles[node]] for node in nxG.nodes]

    # plt.figure()
    # with warnings.catch_warnings():
    #     # catch matplotlib deprecation warning
    #     warnings.simplefilter('ignore')
    #     nx.draw(
    #         nxG,
    #         pos=nx.spring_layout(nxG, seed=42),
    #         with_labels=True,
    #         node_color=node_colors,
    #     )
    # plt.savefig(f"roles_fig/{choice}.png")

    return choice


if __name__ == "__main__":
    processes = []
    os.makedirs("./dataset/roles", exist_ok=True)
    os.makedirs("./dataset/roles_X", exist_ok=True)
    os.makedirs("./dataset/roles_fig", exist_ok=True)
    os.makedirs("./dataset/pos_nx_normalized", exist_ok=True)
    filenames = os.listdir("./dataset/pos_nx0")
    choices = []
    for filename in filenames:
        choices.append(filename.split(".")[0])

    # use multiprocess to speedup
    with Pool(len(choices)) as p:
        print(p.map(f, choices))
    # f(choices[0])
    # f("0xe9f1d62c671efe99896492766c0b416bd3fb9e52")
