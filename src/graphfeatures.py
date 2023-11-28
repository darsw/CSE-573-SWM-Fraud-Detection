
import pandas as pd
import torch
import networkx as nx
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


df1 = pd.read_csv("/data/bs140513_032310.csv")
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

data_reduced.head(5)



df = data_reduced
# New Step: Compute Graph-Based Features
G = nx.Graph()
G.add_edges_from(zip(df['customer'], df['merchant']))

# Compute metrics
betweenness = nx.betweenness_centrality(G)
pagerank = nx.pagerank(G)
degree = dict(G.degree())

# Map metrics to DataFrame
df['betweenness_customer'] = df['customer'].map(betweenness)
df['pagerank_customer'] = df['customer'].map(pagerank)
df['degree_customer'] = df['customer'].map(degree)

df['betweenness_merchant'] = df['merchant'].map(betweenness)
df['pagerank_merchant'] = df['merchant'].map(pagerank)
df['degree_merchant'] = df['merchant'].map(degree)

data_reduced = df

df1 = data_reduced
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler


# Separate features and target variable
X = df1.drop('fraud', axis=1)
y = df1['fraud']
df1.to_csv('graphfeatures.csv', index=False)


