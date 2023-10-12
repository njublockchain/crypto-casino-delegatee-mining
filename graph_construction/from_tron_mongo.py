from concurrent.futures import ThreadPoolExecutor, wait
from io import FileIO
import threading
from pymongo import MongoClient
import json
from web3 import Web3

client = MongoClient("mongodb://localhost:27017/")

db = client["tronrpc"]

tx_cache = set()
set_lock = threading.Lock()

file_lock = threading.Lock()

# TODO: use label db to replace this PASS set
PASS = set(
    [
        "TM1zzNDZD2DPASbKcgdVoTYhfmYgtfwx9R",  # OKEX
        "TV6MuMXfmLbBqPZvBHdwFsDnQeVfnmiuSi",  # Binance-Hot 2"
        "TAzsQ9Gx8eqFNFSKbeXrbi45CuVPHzA8wr",  # "Binance-Hot 5",
        "TQrY8tryqsYVCYS3MFbtffiPp2ccyn4STm",  # "Binance 2",
        "TJDENsfBJs4RFETt1X1W8wMDc8M5XnJhCe",  # Binance-Hot 6",
        "TYASr5UV6HEcXatwdFQfmLVUqQQQMUxHLS",  # "Binance-Hot 3",
        "TNXoiAJ3dct8Fjg4M9fkLFh9S2v9TXc32G",  # Binance-Hot 4",
        "TZHW3PJe6VoyV8NWAaeukqWNRe3bVU7u8n",  # bitget
        "TGZ959FTLRk8droUqDNgLxML1X9mEVej8q",  # bitget
        "TYiQTHtgLo6KX6hYgbKLJsTbWK5hu9X5MG",  # bitget
        "TXFBqBbqJommqZf7BV8NNYzePh97UmJodJ",  # bitfinex
        "TBytnmJqL47n8bAP2NgPWfboXCwEUfEayv",  # bitget
        "TYDzsYUEpvnYmQk4zGP9sWWcTEd2MiAtW6",  # "FTX",
        "TKHuVq1oKVruCGLvqVexFs6dawKv6fQgFs",  # tether
        "TAUN6FwrnwwmaEqYcckffC7wYmbaS6cBiX",  # Binance-Hot 1
        "TQrY8tryqsYVCYS3MFbtffiPp2ccyn4STm",  # Binance 2
        "TN93Yq91SpCtiHaLGNXq8HCJ4WsvRK5dup",  # BinanceAccount
        "TEtESLucSMbU5kqjSYXHQi944jQqWMkfcA",  # binance
        "TMWAaQwcbRukH1bCgYQtfxvUkxrQyk7S9S",  # binance eu
        "TEaKJuXXA8yr61KcGuiQ4TRB9RxHLpXGBW",  # BinanceHot
        "TAAeryFaSds1H8qCWTnfiMfsZkaDGf8mq9",  # BinanceHOTRESERVE7
        "TKYwM2SYNsW79rhP4EqhAZtdP3aXyBNsE2",  # Binance1
        "TGEwJxVErWagXnriZATPMBFFbbeuad9m3h",  # ProBit Exchange
        "TWNxPcZMkLMNHYKeew7FgwoGWjwJFezvCf",  # SatoExchange
        "TRXDEXMoaAprSGJSwKanEUBqfQjvQEDuaw",  # trx dex
        "TXw4rm6zorgEKtQtzpJUgR2rgzxoTyL4xC",  # Pi
        "TJg1igi3WU1LXorqzNJAV6EoJHeSvvyKG5",  # exchangeno09
        "TBShFz6ZKyEySS2vgd4s2yDsCTkQxtfqvy",  # "p2pb2b"
        "TMSr5kHo3LPMQx2YBZeq85ySRiQyVrwKeZ",  # RefCryptoExchange_Bot
        "TGzNdQBmFqisqsbSwEbBunoBegVQsYALRh",  # HaifaExchange
        "TAzJPY5kfDuiDxAgaWrpvRzf7mRiQFfDio",  # Byte_Exchange_GasTank
        "TXFBQPTsCnYMHPH9yKqcDxDYcHtyCXBmhk",  # Arzunexchange
        "TQnysGc4t65XhQYK15Gz4QrnckvCCLXf9D",  # FortExchange
        "TSuAxR6ia8qa5MBPPpPYWXXoDcEHXBQfQ1",  # OCCE_EXCHANGE
        "TPgfTH6GLaFTuuXDs9jkpygSAFHBFeBEfo",  # SEEDexchange
        "TFwCbmyB6vhrbdunQf7RoGMadJHdAaApVX",  # BigSmartDax-Exchange
        "TPNDL2GGtqkmbeJri1tpXkboA6rcrRDNmV",  # Market_Exchange_2
        "TLgkzTdniBn9t5BHsKNFTZtNCYmH7oefEx",  # Alex_TNT_Exchange
        "TQ3ow4M5sCchtputzCQV9gqmbeaqJ87q79",  # DEXExchange-XDX-TelegramEscrow
        "TJuJxDPeqbGT8epGAUcZSuMSxy5SC7aYfT",  # Dholt_City_Exchange
        "TNaRAoLUyYEV2uF7GUrzSjRQTU8v5ZJ5VR",  # Huobi 1
        "TDvf1dSBhR7dEskJs17HxGHheJrjXhiFyM",  # Huobi 2
        "TYPqv3xAmrmgMhBrxncdS4UgvWwoqS6pah",  # huobi_account
        "TEfsGUYxcJf99J4TXkQwL8AWBcdtGfwxxX",  # JustinSun-Huobi
        "TQ9Q3aX8VQ33xv3bCkcxP98dGJM3wvrgDH",  # Huobi-Hot
        "TN2W4cc7a4dsYyTLiLMWa9m7jVpdLjGvYs",  # Huobi_Wallet
        "TXwym1VaATMV1EEPKPmVcZ1oDK8GiB5psy",  # FTX
        "TUN4dKBLbAZjArUS7zYewHwCYA6GSUeSaK",  # Kucoin 3
        "TLWE45u7eusdewSDCjZqUNmyhTUL1NBMzo",  # Kucoin 1
        "TBcUJq55x7Q83ZSr2AqWj59TRj2LvxVr8a",  # Kucoin 2
        "TUpHuDkiCCmwaTZBHZvQdwWzGNm5t8J2b9",  # Kucoin 4
        "TEWzF5ZsaWMh6sTNDPrYaPJrK8TTMGfwCC",  # Kucoin 5
        "TVbVpVtDrARHtCaNMKwc2xYLJQg8rigjXN",  # CoinEx
        "TRz6BJACZuURrcsPeQ7DDhHF2Bp43ehLBM",  # Coinsbit
        "TDoyjmPJHzRFmYfCRLRsPhKjLETwd9fKr9",  # Coinone
        "TVqxFQon6jMkLTdvaXXunvVHPVP4vowf1P",  # Coinex
        "TCYGCdTkY52bFNDLMMaNqYjwB6ELoLecSj",  # Dcoin
        "TWBPGLwQw2EbqYLLw1DJnTDt2ZQ9yJW1JJ",  # WhiteBIT"
        "THQFoJSwtsMMRKCG6B7P5kGcxJQXGi2kiS",  # Bithumb 2
        "TA5vCXk4f1SrCMfz361UxUNABRGP1g2F1r",  # Bittrex 1
    ]
)


def extract_trx_graph_by_hop(f: FileIO, entry_with_41_prefix, hop):
    t_addr = base58.b58encode_check(bytes.fromhex(entry_with_41_prefix)).decode()
    print(f"start for {t_addr} ({hop})")

    next_list = db["transactionColl"].find(
        {
            "raw_data.contract.parameter.value.owner_address": entry_with_41_prefix,
            "raw_data.contract.type": "TransferContract",
            # "ret.contractRet": "SUCCESS",
        },
        {
            "txID": 1,
            "raw_data.contract.parameter.value": 1,
            "raw_data.timestamp": 1,
            "ret.contractRet": 1,
        },
    )
    next_addr_list = set()
    for tx in next_list:
        if tx["ret"][0]["contractRet"] != "SUCCESS":
            print(f"pass {tx}")
            continue
        txID = tx["txID"]
        set_lock.acquire()
        if txID in tx_cache:
            set_lock.release()
            continue
        # else
        tx_cache.add(txID)
        set_lock.release()

        to = tx["raw_data"]["contract"][0]["parameter"]["value"]["to_address"]
        t_to = base58.b58encode_check(bytes.fromhex(to)).decode()
        amount = tx["raw_data"]["contract"][0]["parameter"]["value"]["amount"]
        # ts = tx["raw_data"].get("timestamp")
        # if ts is None:
        ts = db["transactionInfotColl"].find_one({"id": txID})["blockTimeStamp"]

        file_lock.acquire()
        f.write(
            f"{t_addr},{t_to},"
            + json.dumps({"tx": txID, "value": amount, "timestamp": ts})
            + "\n"
        )
        file_lock.release()

        if t_to not in PASS:
            next_addr_list.add(to)
        else:
            print(f"found pass: {t_to}")
    if hop > 0:
        count = len(next_addr_list)
        print(f"{t_addr} has {count} vouts")
        with ThreadPoolExecutor(max_workers=12) as executor:
            fus = [
                executor.submit(extract_trx_graph_by_hop, f, addr, hop - 1)
                for addr in next_addr_list
            ]
            wait(fus)

    prev_list = db["transactionColl"].find(
        {
            "raw_data.contract.parameter.value.to_address": entry_with_41_prefix,
            "raw_data.contract.type": "TransferContract",
            # "ret.contractRet": "SUCCESS",
        },
        {
            "txID": 1,
            "raw_data.contract.parameter.value": 1,
            "raw_data.timestamp": 1,
            "ret.contractRet": 1,
        },
    )
    prev_addr_list = set()
    for tx in prev_list:
        if tx["ret"][0]["contractRet"] != "SUCCESS":
            print(f"pass {tx}")
            continue
        txID = tx["txID"]
        set_lock.acquire()
        if txID in tx_cache:
            set_lock.release()
            continue
        # else
        tx_cache.add(txID)
        set_lock.release()

        frm = tx["raw_data"]["contract"][0]["parameter"]["value"]["owner_address"]
        t_from = base58.b58encode_check(bytes.fromhex(frm)).decode()
        ts = tx["raw_data"]["timestamp"]
        amount = tx["raw_data"]["contract"][0]["parameter"]["value"]["amount"]

        file_lock.acquire()
        f.write(
            f"{t_from},{t_addr},"
            + json.dumps({"tx": txID, "value": amount, "timestamp": ts})
            + "\n"
        )
        file_lock.release()

        if t_from not in PASS:
            prev_addr_list.add(frm)
        else:
            print(f"found pass: {t_from}")
    if hop > 0:
        count = len(prev_addr_list)
        print(f"{t_addr} has {count} vins")
        with ThreadPoolExecutor(max_workers=12) as executor:
            fus = [
                executor.submit(extract_trx_graph_by_hop, f, addr, hop - 1)
                for addr in prev_addr_list
            ]
            wait(fus)


#########################################################################################
USDT_ADDR = "a614f803b6fd780986a42c78ec9c7f77e6ded13c"


def extract_usdt_graph_by_hop(f: FileIO, entry_with_24x0_prefix: str, hop):
    t_addr = base58.b58encode_check(
        bytes.fromhex("41" + entry_with_24x0_prefix.removeprefix("0" * 24))
    ).decode()
    print(f"start for {t_addr} ({hop})")

    next_list = db["transactionInfotColl"].find(
        {
            "log.address": USDT_ADDR,
            "log.topics.0": "ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
            "log.topics.1": entry_with_24x0_prefix,
        }
    )
    next_addr_list = set()
    for tx in next_list:
        txID = tx["id"]
        set_lock.acquire()
        if txID in tx_cache:
            set_lock.release()
            continue
        # else
        tx_cache.add(txID)
        set_lock.release()

        gas = tx["receipt"]["energy_usage_total"]
        ts = tx["blockTimeStamp"]
        for log in tx["log"]:
            if (
                log["topics"][0]
                == "ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
                and log["topics"][1] == entry_with_24x0_prefix
            ):
                to = log["topics"][2]

                t_to = base58.b58encode_check(
                    bytes.fromhex("41" + to.removeprefix("0" * 24))
                ).decode()
                # print(log["data"])
                amount = Web3.to_int(hexstr="0x" + log["data"])
                file_lock.acquire()
                f.write(
                    f"{t_addr},{t_to},"
                    + json.dumps(
                        {"tx": txID, "value": amount, "timestamp": ts, "gas": gas}
                    )
                    + "\n"
                )
                file_lock.release()

                if t_to not in PASS:
                    next_addr_list.add(to)
                else:
                    print(f"found pass: {t_to}")

    if hop > 0:
        count = len(next_addr_list)
        print(f"{t_addr} has {count} vins")
        with ThreadPoolExecutor(max_workers=12) as executor:
            fus = [
                executor.submit(extract_usdt_graph_by_hop, f, addr, hop - 1)
                for addr in next_addr_list
            ]
            wait(fus)

    prev_list = db["transactionColl"].find(
        {
            "log.address": USDT_ADDR,
            "log.topics.0": "ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
            "log.topics.2": entry_with_24x0_prefix,
        },
        {"log": 1, "receipt": 1, "blockTimeStamp": 1},
    )
    prev_addr_list = set()
    for tx in prev_list:
        txID = tx["id"]
        set_lock.acquire()
        if txID in tx_cache:
            set_lock.release()
            continue
        # else
        tx_cache.add(txID)
        set_lock.release()

        gas = tx["receipt"]["energy_usage_total"]
        ts = tx["blockTimeStamp"]
        for log in tx["log"]:
            if (
                log["topics"][0]
                == "ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
                and log["topics"][2] == entry_with_24x0_prefix
            ):
                frm = log["topics"][1]

                t_from = base58.b58encode_check(
                    bytes.fromhex("41" + frm.removeprefix("0" * 24))
                ).decode()
                amount = Web3.to_int(log["data"])
                file_lock.acquire()
                f.write(
                    f"{t_from},{t_addr},"
                    + json.dumps(
                        {"tx": txID, "value": amount, "timestamp": ts, "gas": gas}
                    )
                    + "\n"
                )
                file_lock.release()

                if t_from not in PASS:
                    next_addr_list.add(frm)
                else:
                    print(f"found pass: {t_from}")

    if hop > 0:
        count = len(prev_addr_list)
        print(f"{t_addr} has {count} vins")
        with ThreadPoolExecutor(max_workers=12) as executor:
            fus = [
                executor.submit(extract_usdt_graph_by_hop, f, addr, hop - 1)
                for addr in prev_addr_list
            ]
            wait(fus)

        # for addr in prev_addr_list:
        #     extract_trx_graph_by_hop(f, addr, hop - 1)


#########################################################################################

if __name__ == "__main__":
    import base58

    coin_type = "trx"
    # with open("TAXAYjswtztY7ktYtxyArfwbsjE9JykwWo_trx.edgelist", "w") as f:
    #     extract_trx_graph_by_hop(
    #         f, base58.b58decode_check("TAXAYjswtztY7ktYtxyArfwbsjE9JykwWo").hex(), 3
    #     )
    with open("TAXAYjswtztY7ktYtxyArfwbsjE9JykwWo_usdt.edgelist", "w") as f:
        extract_usdt_graph_by_hop(
            f,
            "0" * 24
            + base58.b58decode_check("TAXAYjswtztY7ktYtxyArfwbsjE9JykwWo")
            .hex()
            .removeprefix("41"),
            3,
        )
