from igraph import Graph

file = "0xc2a81eb482cb4677136d8812cc6db6e0cb580883"

class NodeAnalyzer:
    def __init__(self):
        self.graph = None
        
    def load_graph_from_graphml(self, file_path):
        self.graph = Graph.Read_GraphML(file_path)
        return self.graph

    def add_cluster_attribute(self, cluster_values):
        """ 
        Add 'cluster' attribute to nodes of the graph. 
        cluster_values is a list containing cluster values for each node.
        """
        if not self.graph:
            raise ValueError("Graph has not been loaded!")
        
        if len(cluster_values) != len(self.graph.vs):
            raise ValueError("Length of cluster_values does not match number of nodes in the graph.")
        
        # Add cluster values to the nodes
        self.graph.vs['cluster'] = cluster_values

    def save_graph_as_graphml(self, file_path):
        """ Save the graph back to a GraphML file """
        if not self.graph:
            raise ValueError("Graph has not been loaded!")
            
        self.graph.write_graphml(file_path)

# Example usage:
analyzer = NodeAnalyzer()
analyzer.load_graph_from_graphml(f"./hireachy-data/{file}/{file}_complete.graphml")

# Assuming have a list named clusters with the cluster values
clusters = [2, 2, 2, 1, 5, 5, 4, 5, 5, 3, 4, 5, 5, 3, 4, 3, 5, 2, 2, 2, 2, 2, 2, 0, 2, 0, 0, 0, 1, 0, 5, 2, 4, 2, 2, 2, 2, 0, 1, 5, 5, 2, 2, 1, 5, 1, 3, 5, 1, 5, 1, 0, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 2, 5, 5, 1, 5, 3, 1, 1, 0, 3, 1, 1, 2, 2, 1, 1, 2, 0, 2, 5, 5, 2, 5, 4, 4, 5, 4, 4, 5, 1, 1, 3, 4, 4, 5, 5, 5, 3, 4, 4, 4, 5, 3, 5, 1, 5, 5, 1, 5, 1, 5, 5, 4, 0, 0, 5, 5, 5, 5, 5, 5, 3, 4, 4, 4, 4, 1, 1, 3, 1, 5, 4, 1, 1, 1, 5, 5, 3, 4, 4, 1, 5, 4, 5, 1, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 3, 5, 5, 4, 1, 0, 3, 5, 3, 3, 3, 3, 5, 5, 5, 4, 4, 3, 3, 4, 0, 5, 1, 1, 5, 5, 3, 3, 3, 5, 1, 4, 5, 5, 1, 5, 5, 5, 3, 4, 3, 1, 3, 5, 0, 0, 5, 0, 5, 5, 5, 5, 1, 1, 5, 5, 5, 5, 5, 4, 5, 1, 5, 5, 5, 1, 3, 1, 3, 5, 0, 0, 5, 3, 0, 5, 5, 5, 1, 3, 5, 0, 5, 3, 3, 5, 5, 5, 4, 1, 5, 5, 4, 0, 5, 5, 5, 5, 5, 5, 5, 2, 4, 5, 5, 5, 3, 0, 0, 5, 3, 5, 0, 5, 5, 5, 4] # and so on
analyzer.add_cluster_attribute(clusters)

analyzer.save_graph_as_graphml(f"./hireachy-data/{file}/{file}_cluster.graphml")
