import os
import networkx as nx
from clickhouse_driver import Client

# Initialize the client with the necessary ClickHouse connection parameters
client = Client('localhost')

def get_token_transfers(token_address, address):
    # Query the database for token transfer events involving the given address
    query = """
    SELECT topics, data
    FROM ethereum.events
    WHERE address = %(token_address)s AND
          arrayExists(x -> x = %(address)s, topics) AND
          topics[0] = '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'
    """
    return client.execute(query, {'token_address': token_address, 'address': address})

def parse_transfer_event(event):
    # Parse a token transfer event to extract 'from' and 'to' addresses
    # Assuming the token follows ERC-20 standard and the topics[1] is 'from' and topics[2] is 'to'
    topics = event[0]
    from_address = topics[1][-20:]  # Extract last 20 bytes for address
    to_address = topics[2][-20:]  # Extract last 20 bytes for address
    return from_address, to_address

def n_hop_token_transfers(token_address, address, n):
    graph = nx.DiGraph()
    queue = [(address, 0)]
    seen = {address}

    while queue:
        current_address, current_depth = queue.pop(0)
        if current_depth < n:
            transfers = get_token_transfers(token_address, current_address)
            for event in transfers:
                from_address, to_address = parse_transfer_event(event)
                graph.add_edge(from_address, to_address)
                if to_address not in seen:
                    seen.add(to_address)
                    queue.append((to_address, current_depth + 1))
                if from_address not in seen:
                    seen.add(from_address)
                    queue.append((from_address, current_depth + 1))
    return graph

# Replace with the token contract address and the address you're interested in
token_address = 'your_token_address_here'
address = 'your_address_here'
N = 2  # Example for 2-hop transfers

# Build the N-hop token transfer graph
n_hop_graph = n_hop_token_transfers(token_address, address, N)


import pickle

pickle.dump(n_hop_graph, open(f'graph_{address}_{N}_tf.gexf', "wb"))
