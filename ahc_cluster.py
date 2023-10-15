import os
import igraph as ig
import networkx as nx
from node2vec import Node2Vec
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# List of GraphML files
folder_path = "./graphml"

graphml_files = []

for file_name in os.listdir(folder_path):
    if file_name.endswith(".graphml"):
        file_path = os.path.join(folder_path, file_name)

        # Load graph from GraphML file
        G = ig.Graph.Read_GraphML(file_path)

        # 2. Node representation learning
        node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
        model = node2vec.fit(window=10, min_count=1)

        # 3. Get node vector representations
        node_vectors = {node: model.wv[node] for node in G.nodes()}

        # 4. Read node attributes from GraphML, select needed attributes
        selected_attributes = ["attr1", "attr2", ...]
        node_attributes = {}
        for node in G.nodes():
            attributes = {
                attr: G.nodes[node].get("data", {}).get(attr, 0.0)
                for attr in selected_attributes
            }
            node_attributes[node] = attributes

        # 5. Concatenate node vectors and attributes
        node_features = {}
        for node in G.nodes():
            features = np.concatenate(
                (node_vectors[node], list(node_attributes[node].values()))
            )
            node_features[node] = features

        # 6. Use Silhouette Coefficient to select k
        max_k = 20
        best_k = 2
        best_silhouette_score = -1

        for k in range(2, max_k + 1):
            clustering = AgglomerativeClustering(n_clusters=k)
            labels = clustering.fit_predict(list(node_features.values()))
            silhouette_avg = silhouette_score(list(node_features.values()), labels)

            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg
                best_k = k

        print(f"Best k value based on Silhouette Coefficient: {best_k}")

        # 7. Clustering - Use Agglomerative Clustering
        X = np.array(list(node_features.values()))
        clustering = AgglomerativeClustering(n_clusters=best_k)
        labels = clustering.fit_predict(X)

        # 8. Visualize clustering results
        colors = ["r", "g", "b", "c", "m", "y", "k"]
        color_map = [colors[label % len(colors)] for label in labels]

        # 9. Reduce node vectors to 2D for plotting
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

        # 10. Plot nodes
        plt.figure(figsize=(10, 8))
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=color_map, alpha=0.5)
        plt.title("Node Clustering Visualization")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")

        # Label nodes
        for i, txt in enumerate(G.nodes()):
            plt.annotate(txt, (X_2d[i, 0], X_2d[i, 1]), fontsize=8)

        plt.savefig(f"{file_name}.cluster_result.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Print cluster label for each node
        for node, label in zip(G.nodes(), labels):
            print(f"Node {node}: Cluster {label}")
