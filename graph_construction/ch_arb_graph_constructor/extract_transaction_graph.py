import networkx as nx
import clickhouse_connect
import pickle

client = clickhouse_connect.get_client(
    host="localhost", port=8123, username="default", password=""
)


def get_transactions(address):
    # Query the database for transactions from or to the given address
    addr_no_prefix = address.removeprefix("0x")
    query = f"""
    SELECT hex(`from`) as from_address, hex(`to`) as to_address, value, blockTimestamp, blockNumber
    FROM arbitrumOne.transactions
    WHERE `from` = unhex('{addr_no_prefix}') OR `to` = unhex('{addr_no_prefix}')
    """
    trans = client.query(query).result_rows
    print(f"got {len(trans)} for {address}")
    return trans


def n_hop_neighbours(address, n):
    graph = nx.MultiDiGraph()
    queue = [(address, 0)]
    seen = {address}

    while queue:
        current_address, current_depth = queue.pop(0)
        if current_depth < n:
            transactions = get_transactions(current_address)
            for tx in transactions:
                from_address, to_address, value, blockTimestamp, blockNumber = tx
                if to_address is None:
                    to_address = "0x"
                if from_address[:2] != "0x":
                    from_address = "0x" + from_address
                if to_address[:2] != "0x":
                    to_address = "0x" + to_address

                graph.add_edge(from_address, to_address, value=value, blockTimestamp=blockTimestamp, blockNumber=blockNumber)
                if to_address not in seen:
                    seen.add(to_address)
                    queue.append((to_address, current_depth + 1))
                if from_address not in seen:
                    seen.add(from_address)
                    queue.append((from_address, current_depth + 1))
    return graph


def main():
    client = clickhouse_connect.get_client(
        host="localhost", port=8123, username="default", password=""
    )
    address = "0xc4a482146c2b493066aa7427d23bea4f66e5279c"
    n = 1
    # graph = build_graph_for_address(client, address, N)
    # client = Client('localhost')
    # address = 'your_target_address'
    # n = 5  # for example

    # G = nx.MultiDiGraph()
    G = n_hop_neighbours(address, n)

    # You can now use G for any analysis or visualization purposes provided by NetworkX.
    print(f"Nodes in the graph: {G.nodes(data=True)}")
    print(f"Edges in the graph: {G.edges(data=True)}")

    # save G into a file
    filename = f"graph_{address}_{n}.pkl"
    pickle.dump(G, open(filename, "wb"))


if __name__ == "__main__":
    main()
