# import xml.etree.ElementTree as ET

# def parse_gexf_write_txt(gexf_file_path, txt_file_path):
#     # Parse the XML file
#     tree = ET.parse(gexf_file_path)
#     root = tree.getroot()

#     # Namespaces required to find elements
#     namespaces = {
#         'ns': 'http://www.gexf.net/1.2draft',
#         'viz': 'http://www.gexf.net/1.1draft/viz'
#     }

#     # Open the txt file to write
#     with open(txt_file_path, 'w') as file:
#         file.write("Source\tTarget\tAmount\n")  # Write the header

#         # Iterate over each edge element in the GEXF file
#         for edge in root.findall('.//ns:edges/ns:edge', namespaces):
#             source = edge.get('source')
#             target = edge.get('target')
#             amount = None
            
#             # Find the 'amount' attribute within this edge
#             for attvalue in edge.findall('.//ns:attvalues/ns:attvalue[@for="2"]', namespaces):
#                 amount = attvalue.get('value')

#             # Write to file if amount is found
#             if amount is not None:
#                 file.write(f"{source}\t{target}\t{amount}\n")

# # Specify the paths to your files
# gexf_file_path = '/home/ta/gambling/www-submission/case_dataset/Tron case dataset/Tron_hanhua_graph_1.gexf'
# txt_file_path = '/home/ta/gambling/www-submission/case_dataset/Tron case dataset/Tron_hanhua_graph_1.txt'

# # Run the function
# parse_gexf_write_txt(gexf_file_path, txt_file_path)


import networkx as nx

def convert_gexf_to_txt(gexf_path, txt_path):
    # Load the GEXF graph
    G = nx.read_gexf(gexf_path)

    # Dictionary to hold node labels and assign integers
    node_id_map = {}
    node_counter = 0

    # Open the output file
    with open(txt_path, 'w') as f:
        f.write("Source\tTarget\tAmount\n")  # Header
        # Iterate over the edges
        for edge in G.edges(data=True):
            source = edge[0]
            target = edge[1]
            # Check if the node already has an integer id, if not assign a new one
            if source not in node_id_map:
                node_id_map[source] = node_counter
                node_counter += 1
            if target not in node_id_map:
                node_id_map[target] = node_counter
                node_counter += 1

            # Get amount from the edge attributes, assumed to be stored under 'amount'
            amount = edge[2].get('value', 0)  # Default to 0 if 'amount' is not present

            # Write to file
            f.write(f"{node_id_map[source]}\t{node_id_map[target]}\t{amount}\n")

if __name__ == '__main__':
    gexf_file_path = '/home/ta/gambling/www-submission/case_dataset/Arbitrum case dataset/graph_0xc4a482146c2b493066aa7427d23bea4f66e5279c.gexf'  # Adjust the path to your GEXF file
    txt_file_path = '/home/ta/gambling/www-submission/case_dataset/Arbitrum case dataset/graph_0xc4a482146c2b493066aa7427d23bea4f66e5279c.txt'  # Adjust the path for your output TXT file
    convert_gexf_to_txt(gexf_file_path, txt_file_path)
