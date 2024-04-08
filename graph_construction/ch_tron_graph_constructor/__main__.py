import binascii
import pickle

import networkx as nx
import clickhouse_connect


# client = Client(host='localhost', user='default', password='')
# address = '0x32be343b94f860124dc4fee278fdcbd38c102d88'
# N = 1
# graph = build_graph_for_address(client, address, N)


def add_n_hops_to_graph(G, client, address, n, visited=None):
    if type(address) == bytes:
        address = address.decode("utf-8")

    if visited is None:
        visited = set()

    # Base case
    if n == 0 or address in visited:
        return

    visited.add(address)
    print(address)

    next_address_set = set()

    query = f"""
        SELECT toAddress, transactionHash, blockNum, amount
        FROM tron.transferContracts
        WHERE ownerAddress = '{address}'
        """
    result = client.query(query)
    for result in result.result_rows:
        next_address = result[0]
        transactionHash = result[1]
        blockNum = result[2]
        amount = result[3]

        if type(next_address) == bytes:
            next_address = next_address.decode("utf-8")
        G.add_edge(
            address, next_address, transactionHash=binascii.hexlify(transactionHash), blockNum=blockNum, amount=amount
        )
        next_address_set.add(next_address)

    query = f"""
        SELECT ownerAddress, transactionHash, blockNum, amount
        FROM tron.transferContracts
        WHERE toAddress = '{address}'
        """
    result = client.query(query)
    for result in result.result_rows:
        next_address = result[0]
        transactionHash = result[1]
        blockNum = result[2]
        amount = result[3]

        if type(next_address) == bytes:
            next_address = next_address.decode("utf-8")
        G.add_edge(
            next_address, address, transactionHash=binascii.hexlify(transactionHash), blockNum=blockNum, amount=amount
        )
        next_address_set.add(next_address)

    for next_address in next_address_set:
        add_n_hops_to_graph(G, client, next_address, n - 1, visited)


def main():
    client = clickhouse_connect.get_client(
        host="localhost", port=8123, username="default", password=""
    )
    address = "TY825nrM5GiztWFRQW3JpPUAGuZhWPisSA"
    n = 2
    # graph = build_graph_for_address(client, address, N)
    # client = Client('localhost')
    # address = 'your_target_address'
    # n = 5  # for example

    G = nx.MultiDiGraph()
    add_n_hops_to_graph(G, client, address, n)

    # You can now use G for any analysis or visualization purposes provided by NetworkX.
    print(f"Nodes in the graph: {G.nodes(data=True)}")
    print(f"Edges in the graph: {G.edges(data=True)}")

    # save G into a file
    filename = f"graph_{address}_{n}.pkl"
    pickle.dump(G, open(filename, "wb"))


if __name__ == "__main__":
    main()
