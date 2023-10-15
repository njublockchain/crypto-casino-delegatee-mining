import os
import igraph as ig
import matplotlib.pyplot as plt

# Set folder path
folder_path = './dataset/roles_graphml'

# Initialize metrics dictionary  
metrics = {
  'Density': [],
  'Reciprocity': [], 
  'Transitivity': [],
  'Assortativity': [],
  'Graph Diameter': [],
  'Average Path Length': []
}

# Initialize file names list
file_names = []

# Loop through all files in folder
for file_name in os.listdir(folder_path):
  if file_name.endswith('.graphml'):
    
    # Read graph
    G = ig.load(os.path.join(folder_path, file_name))
    
    # Calculate metrics
    density = G.density(loops=False)
    reciprocity = G.reciprocity(ignore_loops=True)
    transitivity = G.transitivity_avglocal_undirected(mode='zero')
    assortativity = G.assortativity_degree(directed=True)
    diameter = G.diameter(directed=True)
    avg_path_length = G.average_path_length(directed=True)
    
    # Add metrics to dictionary
    metrics['Density'].append(density)
    metrics['Reciprocity'].append(reciprocity) 
    metrics['Transitivity'].append(transitivity)
    metrics['Assortativity'].append(assortativity)
    metrics['Graph Diameter'].append(diameter)
    metrics['Average Path Length'].append(avg_path_length)

    # Add first 5 characters of file name to list
    file_names.append(file_name[:5])

# Generate and save bar charts  
for metric, values in metrics.items():
  plt.bar(file_names, values)
  plt.xlabel('Graph')
  plt.ylabel(metric)
  plt.title(metric)
  plt.xticks(rotation=90, ha='right')
  plt.tight_layout()
  plt.savefig(f'{metric}.png')
  plt.show()
