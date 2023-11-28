import numpy as np
import pandas as pd
import torch
import networkx as nx
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


df1 = pd.read_csv("/content/drive/MyDrive/swm_project/data/bs140513_032310.csv")
from imblearn.over_sampling import RandomOverSampler

# dropping zipcodeori and zipMerchant since they have only one unique value
data_reduced = df1.drop(['zipcodeOri','zipMerchant'],axis=1)
# turning object columns type to categorical for easing the transformation process
col_categorical = data_reduced.select_dtypes(include= ['object']).columns
for col in col_categorical:
    data_reduced[col] = data_reduced[col].astype('category')
# categorical values ==> numeric values
data_reduced[col_categorical] = data_reduced[col_categorical].apply(lambda x: x.cat.codes)

# add transaction_id
data_reduced['transaction_id'] = np.arange(len(data_reduced))+1


df = data_reduced[(data_reduced['customer'] == 2132) | (data_reduced['customer'] == 2767)] 



df = df.sample(n=30)

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'df' is your DataFrame

# Create a graph
G = nx.DiGraph()  # Directed graph, since money flows from customer to merchant

# Add nodes and edges
for index, row in df.iterrows():
    customer_node = f"C{row['customer']}"
    merchant_node = f"M{row['merchant']}"
    transaction_node = f"T{index}"  # Unique transaction node

    # Add nodes with their types
    G.add_node(customer_node, type='customer')
    G.add_node(merchant_node, type='merchant')
    G.add_node(transaction_node, type='transaction', fraud=row['fraud'])

    # Add edges: customer to transaction, transaction to merchant
    G.add_edge(customer_node, transaction_node)
    G.add_edge(transaction_node, merchant_node)

# Define color mapping for nodes
node_color_map = []
for node in G:
    node_type = G.nodes[node]['type']
    if node_type == 'customer':
        node_color_map.append('lightblue')
    elif node_type == 'merchant':
        node_color_map.append('grey')
    else:  # transaction
        node_color_map.append('salmon' if G.nodes[node]['fraud'] else 'lightgreen')

# Define color mapping for edges
edge_color_map = []
for u, v in G.edges:
    if 'fraud' in G.nodes[v]:  # Checking if the node is a transaction
        edge_color_map.append('salmon' if G.nodes[v]['fraud'] else 'green')
    else:
        edge_color_map.append('black')  # Customer to transaction edge

# Set the size of the plot
plt.figure(figsize=(15, 15))

# # Position nodes using a layout
# pos = nx.spring_layout(G)

# # Draw nodes
# nx.draw_networkx_nodes(G, pos, node_color=node_color_map, node_size=100)

# # Draw edges
# nx.draw_networkx_edges(G, pos, arrows=True, edge_color=edge_color_map)

# # Draw labels
# nx.draw_networkx_labels(G, pos)


pos = nx.spring_layout(G, k=1.5, iterations=50)  # k: Optimal distance between nodes, iterations: Number of iterations of force-directed algorithm

# Plot configuration
plt.figure(figsize=(10, 10))
nx.draw_networkx_edges(G, pos, arrows=True, edge_color=edge_color_map, width=1.5, arrowsize = 15)
nx.draw_networkx_nodes(G, pos, node_color=node_color_map, node_size=400)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight=20)
# Show plot
plt.show()
