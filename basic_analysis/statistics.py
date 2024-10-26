import networkx as nx
import pandas as pd

def compute_graph_metrics(graphml_file, output_csv):
    # Load the graph from the GraphML file
    G = nx.read_graphml(graphml_file)

    # Ensure the graph is directed
    if not nx.is_directed(G):
        G = G.to_directed()

    # Calculate in-degree and out-degree
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())

    # Calculate in-weighted-degree and out-weighted-degree (using 'transfer_value' attribute)
    in_weighted_degree = dict(G.in_degree(weight='transfer_value'))
    out_weighted_degree = dict(G.out_degree(weight='transfer_value'))

    # Calculate closeness centrality
    closeness_centrality = nx.closeness_centrality(G)

    # Calculate betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(G)

    # Calculate hubness and authoritativeness using HITS algorithm
    hits_hubness, hits_authority = nx.hits(G)

    # Calculate PageRank
    pagerank = nx.pagerank(G)

    # Calculate eigenvector centrality
    # eigenvector_centrality = nx.eigenvector_centrality(G)

    # Prepare the data for export
    data = {
        'Node': list(G.nodes),
        'In-Degree': [in_degree[node] for node in G.nodes],
        'Out-Degree': [out_degree[node] for node in G.nodes],
        'In-Weighted-Degree': [in_weighted_degree[node] for node in G.nodes],
        'Out-Weighted-Degree': [out_weighted_degree[node] for node in G.nodes],
        'Closeness Centrality': [closeness_centrality[node] for node in G.nodes],
        'Betweenness Centrality': [betweenness_centrality[node] for node in G.nodes],
        'Hubness': [hits_hubness[node] for node in G.nodes],
        'Authoritativeness': [hits_authority[node] for node in G.nodes],
        'PageRank': [pagerank[node] for node in G.nodes],
        # 'Eigenvector Centrality': [eigenvector_centrality[node] for node in G.nodes]
    }

    # Create a DataFrame and save it as a CSV file
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Metrics successfully saved to {output_csv}")

if __name__ == '__main__':
    # Specify the input GraphML file and output CSV file
    graphml_file = '/home/lab0/gnn-wjx/0xc2a81eb482cb4677136d8812cc6db6e0cb580883.graphml'
    output_csv = '/home/lab0/gnn-wjx/graph_metrics.csv'
    
    # Compute the graph metrics and save to CSV
    compute_graph_metrics(graphml_file, output_csv)
