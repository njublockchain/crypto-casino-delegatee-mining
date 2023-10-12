import pickle
import tqdm
from builtins import float, len, open, set
import numpy as np
import os
from pymongo import MongoClient
import pandas as pd
import networkx as nx

TOKEN_TRANSFER = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"


def gini(x, w=None):
    # The rest of the code requires numpy arrays.
    if len(x) == 0:
        return np.nan

    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / (
            cumxw[-1] * cumw[-1]
        )
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


class AddressFeatureSet:
    # addr = ""
    sum_val_in = 0.0
    sum_val_out = 0.0

    avg_val_in = 0.0
    avg_val_out = 0.0

    count = 0.0
    count_in = 0.0
    count_out = 0.0

    freq = 0.0
    freq_in = 0.0
    freq_out = 0.0

    gini_val = 0.0
    gini_val_in = 0.0
    gini_val_out = 0.0

    avg_gas = 0.0
    avg_gas_in = 0.0
    avg_gas_out = 0.0

    avg_gasprice = 0.0
    avg_gasprice_in = 0.0
    avg_gasprice_out = 0.0

    in_out_rate = 0.0

    def __init__(
        self,
        # addr,
        val_in_list,
        val_out_list,
        height_in_list,
        height_out_list,
        gas_in_list,
        gas_out_list,
        gasprice_in_list,
        gasprice_out_list,
    ):
        # self.addr = addr
        self.sum_val_in = np.sum(val_in_list)
        self.sum_val_out = np.sum(val_out_list)

        self.avg_val_in = np.average(val_in_list)
        self.avg_val_out = np.average(val_out_list)

        self.count_in = len(val_in_list)
        self.count_out = len(val_out_list)
        self.count = self.count_in + self.count_out

        max_height_in = np.max(height_in_list, initial=0)
        min_height_in = np.min(height_in_list, initial=0)
        max_height_out = np.max(height_out_list, initial=0)
        min_height_out = np.min(height_out_list, initial=0)

        # has at least one in or out
        max_height = np.max(height_in_list + height_out_list, initial=0)
        min_height = np.min(height_in_list + height_out_list, initial=0)

        interval = max_height - min_height
        interval_in = max_height_in - min_height_in
        interval_out = max_height_out - min_height_out

        self.freq = np.nan if interval == 0 else (self.count / interval)
        self.freq_in = np.nan if interval_in == 0 else (self.count_in / interval_in)
        self.freq_out = np.nan if interval_out == 0 else (self.count_out / interval_out)

        val_list = val_in_list + val_out_list

        self.gini_val = gini(val_list)
        self.gini_val_in = gini(val_in_list)
        self.gini_val_out = gini(val_in_list)

        sum_gas_in = np.sum(gas_in_list)
        sum_gas_out = np.sum(gas_out_list)

        self.avg_gas = (sum_gas_in + sum_gas_out) / (self.count_in + self.count_out)
        self.avg_gas_in = np.nan if self.count_in == 0 else (sum_gas_in / self.count_in)
        self.avg_gas_out = (
            np.nan if self.count_out == 0 else (sum_gas_out / self.count_out)
        )

        sum_gasprice_in = np.sum(gasprice_in_list)
        sum_gasprice_out = np.sum(gasprice_out_list)

        self.avg_gasprice = (sum_gasprice_in + sum_gasprice_out) / (
            self.count_in + self.count_out
        )
        self.avg_gasprice_in = (
            np.nan if self.count_in == 0 else (sum_gasprice_in / self.count_in)
        )
        self.avg_gasprice_out = (
            np.nan if self.count_out == 0 else (sum_gasprice_out / self.count_out)
        )

        self.in_out_rate = 0 if self.count_out == 0 else self.count_in / self.count_out


def run_on_token_graph(mgo: MongoClient, path: str, token_contract: str):
    db = mgo["ethrpc"]
    coll = db["receiptColl"]

    nxG: nx.MultiDiGraph = pickle.load(
        open(os.path.join(path, token_contract + ".pkl"), "rb")
    )
    dic = {}

    for addr in tqdm.tqdm(nxG.nodes):
        val_in_list = []
        val_out_list = []
        height_in_list = []
        height_out_list = []
        gas_in_list = []
        gas_out_list = []
        gasprice_in_list = []
        gasprice_out_list = []

        in_tfs = nxG.in_edges(nbunch=addr, data=True)
        out_tfs = nxG.out_edges(nbunch=addr, data=True)

        for _from, _to, in_tf in in_tfs:
            val_in_list.append(float(in_tf["transfer_value"]))
            height_in_list.append(float(in_tf["block_number"]))
            gas_in_list.append(float(in_tf["gas_used"]))
            gasprice_in_list.append(float(in_tf["effective_gas_price"]))

        for _from, _to, out_tf in out_tfs:
            val_out_list.append(float(out_tf["transfer_value"]))
            height_out_list.append(float(out_tf["block_number"]))
            gas_out_list.append(float(out_tf["gas_used"]))
            gasprice_out_list.append(float(out_tf["effective_gas_price"]))

        coll.find({"logs.address": token_contract, "logs.topics.0": TOKEN_TRANSFER})

        feat = AddressFeatureSet(
            # addr,
            val_in_list,
            val_out_list,
            height_in_list,
            height_out_list,
            gas_in_list,
            gas_out_list,
            gasprice_in_list,
            gasprice_out_list,
        )
        record = vars(feat)
        dic[addr] = record
    nx.set_node_attributes(nxG, dic)
    with open(
        os.path.join(path.split("_")[0] + "_nx0", token_contract + ".pkl"), "wb"
    ) as f:
        pickle.dump(nxG, f)

    pd.DataFrame.from_dict(dict(nxG.nodes(data=True), orient="index")).transpose().to_csv(
        os.path.join(path.split("_")[0] + "_pd0", token_contract + ".csv")
    )


if __name__ == "__main__":
    client = MongoClient("mongodb://localhost:27017/")

    from pathlib import Path

    addr_type = "pos"
    Path(f"./{addr_type}_nx0").mkdir(parents=True, exist_ok=True)
    Path(f"./{addr_type}_pd0").mkdir(parents=True, exist_ok=True)

    for filename in tqdm.tqdm(os.listdir(f"./{addr_type}_nx")):
        contract = filename.split(".")[0]
        run_on_token_graph(client, f"./{addr_type}_nx", contract)
