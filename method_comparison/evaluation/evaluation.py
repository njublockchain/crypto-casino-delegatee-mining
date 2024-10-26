# -----------Ethereum
# import pandas as pd
# from sklearn.metrics import accuracy_score
# from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
# import numpy as np

# # Define the data
# data = {
#     "ground_truth": [
#         "Stakeholder", "Stakeholder", "Stakeholder", "Stakeholder", "Stakeholder", 
#         "Stakeholder", "Exchange", "Exchange", "Exchange", "Exchange", 
#         "Stakeholder", "Exchange", "Exchange", "Exchange", "Exchange", "Fund sponsors"
#     ],
#     "predict_ours": [
#         "Stakeholder", "Stakeholder", "Stakeholder", "Stakeholder", "Stakeholder", 
#         "Stakeholder", "Exchange", "gamblers", "Exchange", "Exchange", 
#         "Stakeholder", "Exchange", "Exchange", "listing", "gamblers", "Fund sponsors"
#     ],
#     "SuperGAT+KMeans": [
#         "Stakeholder", "Stakeholder", "Stakeholder", "Stakeholder", "Stakeholder", 
#         "Stakeholder", "Exchange", "Exchange", "Exchange", "Exchange", 
#         "airdrop", "gamblers", "gamblers", "Exchange", "gamblers", "Stakeholder"
#     ],
#     "EGC+KMeans": [
#         "gamblers", "Exchange", "Stakeholder", "Stakeholder", "Exchange", 
#         "Stakeholder", "Exchange", "gamblers", "gamblers", "gamblers", 
#         "gamblers", "gamblers", "gamblers", "gamblers", "gamblers", "gamblers"
#     ],
#     "GATv2+KMeans": [
#         "Stakeholder", "Stakeholder", "Stakeholder", "Stakeholder", "Stakeholder", 
#         "Stakeholder", "Exchange", "gamblers", "Exchange", "gamblers", 
#         "Stakeholder", "Exchange", "Exchange", "arb", "gamblers", "airdrop"
#     ],
#     "GNN-FiLM+KMeans": [
#         "Stakeholder", "Stakeholder", "Stakeholder", "Stakeholder", "airdrop", 
#         "Stakeholder", "Exchange", "gamblers", "Exchange", "Exchange", 
#         "airdrop", "gamblers", "gamblers", "gamblers", "gamblers", "Stakeholder"
#     ],
#     "DirGNN+KMeans": [
#         "Stakeholder", "Stakeholder", "Stakeholder", "airdrop", "airdrop", 
#         "airdrop", "Exchange", "gamblers", "Exchange", "Exchange", 
#         "Stakeholder", "gamblers", "gamblers", "gamblers", "gamblers", "Stakeholder"
#     ],
#     "SuperGAT+DBSCAN": [
#         "Stakeholder", "Stakeholder", "Stakeholder", "Stakeholder", "Stakeholder", 
#         "Stakeholder", "Exchange", "gamblers", "Exchange", "arb", 
#         "Stakeholder", "arb", "Exchange", "arb", "gamblers", "Stakeholder"
#     ],
#     "EGC+DBSCAN": [
#         "Stakeholder", "Stakeholder", "Exchange", "Fund sponsors", "airdrop", 
#         "Fund sponsors", "Exchange", "gamblers", "Exchange", "Exchange", 
#         "airdrop", "gamblers", "gamblers", "gamblers", "gamblers", "airdrop"
#     ],
#     "GATv2+DBSCAN": [
#         "Stakeholder", "Stakeholder", "Stakeholder", "Stakeholder", "Stakeholder", 
#         "Stakeholder", "Exchange", "gamblers", "Exchange", "Exchange", 
#         "Stakeholder", "Exchange", "Exchange", "gamblers", "arb", "Stakeholder"
#     ],
#     "GNN-FiLM+DBSCAN": [
#         "Stakeholder", "Stakeholder", "Stakeholder", "airdrop", "airdrop", 
#         "gamblers", "Exchange", "gamblers", "Exchange", "Exchange", 
#         "airdrop", "gamblers", "gamblers", "gamblers", "gamblers", "Fund sponsors"
#     ],
#     "DirGNN+DBSCAN": [
#         "Stakeholder", "airdrop", "Stakeholder", "Stakeholder", "Stakeholder", 
#         "Stakeholder", "Exchange", "gamblers", "Exchange", "Exchange", 
#         "Stakeholder", "gamblers", "gamblers", "gamblers", "gamblers", "Stakeholder"
#     ],
#     "RolX": [
#         "Exchange", "Stakeholder", "Stakeholder", "arb", "Exchange", 
#         "Stakeholder", "Exchange", "Exchange", "Exchange", "Exchange", 
#         "Exchange", "Exchange", "Exchange", "gamblers", "Exchange", "gamblers"
#     ],
#     "EPLCD": [
#         "Stakeholder", "listing", "Stakeholder", "Stakeholder", "Stakeholder", 
#         "Stakeholder", "Exchange", "gamblers", "gamblers", "Exchange", 
#         "Stakeholder", "gamblers", "arb", "liting", "gamblers", "Stakeholder"
#     ]
# }

# # Create a DataFrame
# df = pd.DataFrame(data)

# # Define a function to calculate purity
# def purity_score(y_true, y_pred):
#     contingency_matrix = np.zeros((len(set(y_true)), len(set(y_pred))))
#     label_mapping_true = {label: i for i, label in enumerate(set(y_true))}
#     label_mapping_pred = {label: i for i, label in enumerate(set(y_pred))}
    
#     for true, pred in zip(y_true, y_pred):
#         contingency_matrix[label_mapping_true[true], label_mapping_pred[pred]] += 1
    
#     return np.sum(np.amax(contingency_matrix, axis=1)) / np.sum(contingency_matrix)

# # Prepare a results DataFrame to store metrics
# results = pd.DataFrame(columns=["Algorithm", "Purity", "Accuracy", "NMI"])

# # Iterate over each column (algorithm's prediction) and calculate metrics
# for algorithm in df.columns[1:]:
#     purity = purity_score(df["ground_truth"], df[algorithm])
#     accuracy = accuracy_score(df["ground_truth"], df[algorithm])
#     nmi_value = nmi(df["ground_truth"], df[algorithm])
    
#     # Use pd.concat instead of append
#     results = pd.concat([results, pd.DataFrame({
#         "Algorithm": [algorithm],
#         "Purity": [purity],
#         "Accuracy": [accuracy],
#         "NMI": [nmi_value]
#     })], ignore_index=True)

# # Display the calculated results
# print(results)

# -----------TRON

# import pandas as pd
# from sklearn.metrics import accuracy_score
# from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
# import numpy as np

# # Data provided in the table
# data = {
#     "ground_truth": [
#         "bonus payers", "bonus payers", "bonus payers", "Fund sponsor", 
#         "Exchange", "Exchange", "Exchange", "Exchange", "Exchange", "Exchange",
#         "gamblers", "gamblers", "disguised gamblers", "Exchange", "Exchange", "disguised gamblers","Betting"
#     ],
#     "ours": [
#         "bonus payers", "bonus payers", "bonus payers", "Fund sponsor", 
#         "Exchange", "Exchange", "Exchange", "Exchange", "Exchange", "Exchange",
#         "gamblers", "gamblers", "gamblers", "gamblers", "gamblers", "disguised gamblers","Betting"
#     ],
#     "SuperGAT+KMeans": [
#         "bonus payers", "bonus payers", "Fund sponsor", "Fund sponsor", 
#         "Exchange", "Exchange", "Exchange", "gamblers", "gamblers", "gamblers",
#         "gamblers", "gamblers", "gamblers", "Exchange", "gamblers", "gamblers","Betting"
#     ],
#     "EGC+KMeans": [
#         "bonus payers", "Fund sponsor", "bonus payers", "bonus payers", 
#         "Exchange", "Fund sponsor", "Exchange", "Fund sponsor", "Exchange", "gamblers",
#         "gamblers", "gamblers", "gamblers", "gamblers", "gamblers", "gamblers","Betting"
#     ],
#     "GATv2+KMeans": [
#         "bonus payers", "bonus payers", "bonus payers", "Fund sponsor", 
#         "Exchange", "Exchange", "Exchange", "Exchange", "gamblers", "gamblers",
#         "gamblers", "gamblers", "gamblers", "gamblers", "gamblers", "gamblers","Betting"
#     ],
#     "GNN-FiLM+KMeans": [
#         "bonus payers", "bonus payers", "bonus payers", "Fund sponsor", 
#         "Exchange", "Exchange", "Exchange", "gamblers", "gamblers", "gamblers",
#         "gamblers", "gamblers", "gamblers", "Fund sponsor", "Exchange", "gamblers","Betting"
#     ],
#     "DirGNN+Kmeans": [
#         "bonus payers", "bonus payers", "bonus payers", "Fund sponsor", 
#         "Exchange", "Exchange", "Betting", "Betting", "gamblers", "Exchange",
#         "gamblers", "gamblers", "gamblers", "Fund sponsor", "Fund sponsor", "gamblers","Betting"
#     ],
#     "SuperGAT+DBSCAN": [
#         "bonus payers", "bonus payers", "bonus payers", "bonus payers", 
#         "Exchange", "Exchange", "Exchange", "gamblers", "gamblers", "Exchange",
#         "gamblers", "gamblers", "gamblers", "gamblers", "gamblers", "disguised gamblers","Betting"
#     ],
#     "EGC+DBSCAN": [
#         "bonus payers", "bonus payers", "bonus payers", "bonus payers", 
#         "Exchange", "Exchange", "Exchange", "gamblers", "gamblers", "Exchange",
#         "gamblers", "gamblers", "gamblers", "gamblers", "gamblers", "gamblers","Betting"
#     ],
#     "GATv2+DBSCAN": [
#         "bonus payers", "bonus payers", "Fund sponsor", "Fund sponsor", 
#         "Exchange", "Exchange", "Exchange", "Exchange", "Betting", "Betting",
#         "gamblers", "gamblers", "disguised gamblers", "Exchange", "gamblers", "disguised gamblers","Betting"
#     ],
#     "GNN-FiLM+DBSCAN": [
#         "bonus payers", "bonus payers", "Fund sponsor", "Fund sponsor", 
#         "Exchange", "Exchange", "Exchange", "gamblers", "Betting", "Betting",
#         "gamblers", "gamblers", "gamblers", "gamblers", "gamblers", "gamblers","Exchange"
#     ],
#     "DirGNN+DBSCAN": [
#         "bonus payers", "bonus payers", "Fund sponsor", "Fund sponsor", 
#         "Exchange", "gamblers", "Exchange", "gamblers", "Betting", "Betting",
#         "gamblers", "gamblers", "disguised gamblers", "gamblers", "gamblers", "disguised gamblers","Betting"
#     ],
#     "RolX": [
#         "bonus payers", "Fund sponsor", "Fund sponsor", "Fund sponsor", 
#         "Exchange", "Exchange", "Exchange", "gamblers", "gamblers", "Exchange",
#         "gamblers", "gamblers", "Betting", "Betting", "gamblers", "gamblers","Betting"
#     ],
#     "EPLCD": [
#         "bonus payers", "bonus payers", "bonus payers", "Fund sponsor", 
#         "Exchange", "Exchange", "Exchange", "gamblers", "gamblers", "gamblers",
#         "gamblers", "gamblers", "gamblers", "gamblers", "Betting", "gamblers","Betting"
#     ]
# }

# df = pd.DataFrame(data)

# # Define a function to calculate purity
# def purity_score(y_true, y_pred):
#     contingency_matrix = np.zeros((len(set(y_true)), len(set(y_pred))))
#     label_mapping_true = {label: i for i, label in enumerate(set(y_true))}
#     label_mapping_pred = {label: i for i, label in enumerate(set(y_pred))}
    
#     for true, pred in zip(y_true, y_pred):
#         contingency_matrix[label_mapping_true[true], label_mapping_pred[pred]] += 1
    
#     return np.sum(np.amax(contingency_matrix, axis=1)) / np.sum(contingency_matrix)

# # Prepare a results DataFrame to store metrics
# results = pd.DataFrame(columns=["Algorithm", "Purity", "Accuracy", "NMI"])

# # Iterate over each column (algorithm's prediction) and calculate metrics
# for algorithm in df.columns[1:]:
#     purity = purity_score(df["ground_truth"], df[algorithm])
#     accuracy = accuracy_score(df["ground_truth"], df[algorithm])
#     nmi_value = nmi(df["ground_truth"], df[algorithm])
    
#     # Append the results
#     results = pd.concat([results, pd.DataFrame({
#         "Algorithm": [algorithm],
#         "Purity": [purity],
#         "Accuracy": [accuracy],
#         "NMI": [nmi_value]
#     })], ignore_index=True)

# # Display the calculated results
# print(results)


# -----------Arbitrum
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import numpy as np

# Data provided in the table
data = {
    "ground_truth": [
        "Exchange", "Exchange", "gamblers", "gamblers", "crossbridge", "crossbridge", 
        "liquidity", "liquidity", "liquidity", "liquidity", "liquidity", "liquidity", 
        "stakeholder"
    ],
    "ours": [
        "Exchange", "Exchange", "gamblers", "gamblers", "crossbridge", "crossbridge", 
        "liquidity", "liquidity", "gamblers", "gamblers", "liquidity", "gamblers", 
        "stakeholder"
    ],
    "SuperGAT+KMeans": [
        "Exchange", "Exchange", "gamblers", "gamblers", "Exchange", "Exchange", 
        "Exchange", "Exchange", "gamblers", "gamblers", "liquidity", "liquidity", 
        "stakeholder"
    ],
    "EGC+KMeans": [
        "Exchange", "gamblers", "Exchange", "gamblers", "Exchange", "Exchange", 
        "gamblers", "liquidity", "liquidity", "gamblers", "liquidity", "gamblers", 
        "stakeholder"
    ],
    "GATv2+KMeans": [
        "Exchange", "Exchange", "Exchange", "Exchange", "Exchange", "Exchange", 
        "gamblers", "liquidity", "liquidity", "gamblers", "liquidity", "gamblers", 
        "stakeholder"
    ],
    "GNN-FiLM+KMeans": [
        "Exchange", "Exchange", "gamblers", "Exchange", "gamblers", "crossbridge", 
        "Exchange", "Exchange", "liquidity", "gamblers", "liquidity", "gamblers", 
        "stakeholder"
    ],
    "DirGNN+Kmeans": [
        "Exchange", "Exchange", "Exchange", "Exchange", "crossbridge", "Exchange", 
        "liquidity", "Exchange", "liquidity", "liquidity", "liquidity", "gamblers", 
        "stakeholder"
    ],
    "SuperGAT+DBSCAN": [
        "Exchange", "Exchange", "gamblers", "Exchange", "crossbridge", "gamblers", 
        "liquidity", "liquidity", "gamblers", "Exchange", "liquidity", "gamblers", 
        "stakeholder"
    ],
    "EGC+DBSCAN": [
        "Exchange", "Exchange", "gamblers", "Exchange", "Exchange", "Exchange", 
        "liquidity", "Exchange", "liquidity", "Exchange", "Exchange", "gamblers", 
        "stakeholder"
    ],
    "GATv2+DBSCAN": [
        "Exchange", "gamblers", "gamblers", "Exchange", "crossbridge", "crossbridge", 
        "liquidity", "Exchange", "liquidity", "Exchange", "liquidity", "Exchange", 
        "stakeholder"
    ],
    "GNN-FiLM+DBSCAN": [
        "Exchange", "Exchange", "gamblers", "Exchange", "crossbridge", "crossbridge", 
        "Exchange", "Exchange", "liquidity", "Exchange", "liquidity", "gamblers", 
        "stakeholder"
    ],
    "DirGNN+DBSCAN": [
        "Exchange", "Exchange", "gamblers", "crossbridge", "gamblers", "Exchange", 
        "liquidity", "liquidity", "liquidity", "liquidity", "liquidity", "stakeholder", 
        "stakeholder"
    ],
    "RolX": [
        "Exchange", "Exchange", "gamblers", "Exchange", "Exchange", "Exchange", 
        "Exchange", "liquidity", "gamblers", "Exchange", "liquidity", "gamblers", 
        "stakeholder"
    ],
    "EPLCD": [
        "Exchange", "Exchange", "gamblers", "Exchange", "crossbridge", "crossbridge", 
        "liquidity", "liquidity", "gamblers", "Exchange", "gamblers", "liquidity", 
        "stakeholder"
    ]
}

df = pd.DataFrame(data)

# Define a function to calculate purity
def purity_score(y_true, y_pred):
    contingency_matrix = np.zeros((len(set(y_true)), len(set(y_pred))))
    label_mapping_true = {label: i for i, label in enumerate(set(y_true))}
    label_mapping_pred = {label: i for i, label in enumerate(set(y_pred))}
    
    for true, pred in zip(y_true, y_pred):
        contingency_matrix[label_mapping_true[true], label_mapping_pred[pred]] += 1
    
    return np.sum(np.amax(contingency_matrix, axis=1)) / np.sum(contingency_matrix)

# Prepare a results DataFrame to store metrics
results = pd.DataFrame(columns=["Algorithm", "Purity","Accuracy", "NMI"])

# Iterate over each column (algorithm's prediction) and calculate metrics
for algorithm in df.columns[1:]:
    purity = purity_score(df["ground_truth"], df[algorithm])
    accuracy = accuracy_score(df["ground_truth"], df[algorithm])
    nmi_value = nmi(df["ground_truth"], df[algorithm])
    
    # Append the results
    results = pd.concat([results, pd.DataFrame({
        "Algorithm": [algorithm],
        "Purity": [purity],
        "Accuracy": [accuracy],
        "NMI": [nmi_value]
    })], ignore_index=True)

# Display the calculated results
print(results)
