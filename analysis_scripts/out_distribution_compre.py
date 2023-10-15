import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import igraph as ig

# Define function for power-law
def power_law(x, a, b):
    return a * np.power(x, b)

# List of GraphML files
folder_path = './roles_graphml'

graphml_files = []

for file_name in os.listdir(folder_path):
  if file_name.endswith('.graphml'):
    file_path = os.path.join(folder_path, file_name)
    graphml_files.append(file_path)


# List of colors for each plot
colors = ['b', 'g', 'r', 'c', 'm', 'y']

plt.figure()

# Loop through each GraphML file and plot the in-degree distribution
for i, graphml_file in enumerate(graphml_files):
    # Load graph from GraphML file
    graph = ig.Graph.Read_GraphML(graphml_file)

    # Compute in-degree of each node
    out_degrees = graph.outdegree()

    # Compute in-degree distribution
    out_degree_distribution = {}
    for deg in out_degrees:
        if deg in out_degree_distribution:
            out_degree_distribution[deg] += 1
        else:
            out_degree_distribution[deg] = 1

    x = np.array(list(out_degree_distribution.keys()))
    y = np.array(list(out_degree_distribution.values()))

    # Filter out zeros from x
    non_zero_indices = x != 0
    x = x[non_zero_indices]
    y = y[non_zero_indices]

    # Fit power-law
    params_power_law, _ = curve_fit(power_law, x, y)

    # Plot in-degree distribution on log-log scale
    plt.loglog(x, y, 'o', color=colors[i])
    plt.loglog(x, power_law(x, *params_power_law), '-', color=colors[i])

plt.xlabel('Out-Degree (log scale)')
plt.ylabel('Frequency (log scale)')
plt.title('Out-Degree Distribution of Six Tokens')
plt.ylim(0.5, plt.ylim()[1])
plt.legend(['FunFair', 'Fit1', 'Virtue Player Points', 'Fit2', 'Herocoin', 'Fit3', 'Cai Token', 'Fit4', 'BispexToken', 'Fit5', 'QNTU Token', 'Fit6'])
plt.savefig('loglog_out_degree_distribution.png')
plt.show()
