import numpy as np
import pandas as pd
from py2neo import Graph
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.decomposition import PCA

# Functions to fetch network-based features
def load_degree(record):
    return records[record.split("'")[1]]['degree']

def load_community(record):
    return str(records[record.split("'")[1]]['community'])

def load_pagerank(record):
    return records[record.split("'")[1]]['pagerank']

banksim_df = pd.read_csv("../data/bs140513_032310.csv")

# Retrieving the class attribute from the dataframe
labels = banksim_df['fraud']


# Connecting to the Neo4j database
graph = Graph(password="MyPassword")

# Query to fetch the network features from Neo4j
query = """
MATCH (p:LINK)
RETURN p.id AS id, p.pagerank as pagerank, p.degree as degree, p.community as community, p.betweenness as betweenness;
"""

data = graph.run(query)

records = {}

for record in data:
    records[record['id']] = {'degree': record['degree'], 'pagerank': record['pagerank'], 'community': record['community']}

# Merging the graph features with the banksim dataset
banksim_df['merchant_degree'] = banksim_df['merchant'].apply(load_degree)
banksim_df['customer_degree'] = banksim_df['customer'].apply(load_degree)
banksim_df['merchant_pagerank'] = banksim_df['merchant'].apply(load_pagerank)
banksim_df['customer_pagerank'] = banksim_df['customer'].apply(load_pagerank)
banksim_df['merchant_community'] = banksim_df['merchant'].apply(load_community)
banksim_df['customer_community'] = banksim_df['customer'].apply(load_community)

# Export dataset with graph features 
banksim_df.to_csv("graph_features_dataset.csv")

