import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import igraph as ig

for file_name in os.listdir(folder_path):
  if file_name.endswith('.graphml'):
    file_path = os.path.join(folder_path, file_name)
    # Load graph from GraphML file
    graph = ig.Graph.Read_GraphML(file_path)

    # Compute in-degree of each node
    out_degrees = graph.outdegree()

    # Compute in-degree distribution
    out_degree_distribution = {}
    for deg in out_degrees:
        if deg in out_degree_distribution:
            out_degree_distribution[deg] += 1
        else:
            out_degree_distribution[deg] = 1

    # Define function for power-law
    def power_law(x, a, b):
        return a * np.power(x, b)

    x = np.array(list(out_degree_distribution.keys()))
    y = np.array(list(out_degree_distribution.values()))

    # Filter out zeros from x
    non_zero_indices = x != 0
    x = x[non_zero_indices]
    y = y[non_zero_indices]

    # Fit power-law
    params_power_law, _ = curve_fit(power_law, x, y)

    # Plot in-degree distribution on log-log scale
    plt.loglog(x, y, 'bo')
    plt.loglog(x, power_law(x, *params_power_law), 'r-')
    plt.xlabel('Out-Degree (log scale)')
    plt.ylabel('Frequency (log scale)')
    plt.title('Out-Degree Distribution')
    plt.ylim(0.5, max(y))
    plt.legend(['Data', 'Power-law fit: a=%5.3f, b=%5.3f' % tuple(params_power_law)])
    plt.savefig(f'{file_name}-loglog_out_degree_distribution.png')
    plt.show()
