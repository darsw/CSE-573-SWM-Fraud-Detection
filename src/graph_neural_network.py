
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.loader import DataLoader

import pandas as pd
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
df1 = pd.read_csv("/content/drive/MyDrive/swm_project/data/graphfeatures.csv")

transaction_features = torch.tensor(df1[['amount', 'transaction_id']].values, dtype=torch.float)
customer_features = torch.tensor(df1[['gender', 'customer']].values, dtype=torch.float)
merchant_features = torch.tensor(df1[['category', 'merchant']].values, dtype=torch.float)
node_features = torch.cat((transaction_features, customer_features, merchant_features), dim=0)

# Define edge indices
num_customers = df1['customer'].nunique()
num_merchants = df1['merchant'].nunique()
customer_indices = torch.tensor(df1['customer'].values, dtype=torch.long)
transaction_indices = torch.tensor(df1.index.values, dtype=torch.long)
merchant_indices = torch.tensor(df1['merchant'].values, dtype=torch.long) + num_customers
edge_index_customer_to_transaction = torch.stack([customer_indices, transaction_indices], dim=0)
edge_index_transaction_to_merchant = torch.stack([transaction_indices, merchant_indices], dim=0)
edge_index = torch.cat((edge_index_customer_to_transaction, edge_index_transaction_to_merchant), dim=1)

# Define edge and node features
columns_to_remove = ['amount', 'gender', 'category', 'transaction_id', 'customer', 'merchant']
df_edge_features = df1.drop(columns=columns_to_remove)
edge_features = torch.tensor(df_edge_features.values, dtype=torch.float)
labels = torch.tensor(df1['fraud'].values, dtype=torch.long)

# Create graph data
graph_data = Data(x=node_features, edge_index=edge_index, y=labels, edge_attr=edge_features)

# DataLoader for graph data
data_loader = DataLoader([graph_data], batch_size=32, shuffle=True)

# Define GNN model
class TransactionGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes):
        super(TransactionGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)  

# Prepare for training and evaluation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
graph_data = graph_data.to(device)
num_transactions = graph_data.y.shape[0]

# Create train and test masks
num_total_nodes = graph_data.num_nodes
train_mask = torch.zeros(num_total_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_total_nodes, dtype=torch.bool)

# Assuming transaction nodes are a contiguous block starting from an offset
transaction_node_start = num_customers + num_merchants  # Adjust this based on your graph structure

# Create train and test indices for transactions
train_indices, test_indices = train_test_split(range(num_transactions), test_size=0.2, random_state=42)

# Mark relevant nodes in the masks
train_mask[transaction_node_start + torch.tensor(train_indices)] = True
test_mask[transaction_node_start + torch.tensor(test_indices)] = True

# Assign masks to graph data
graph_data.train_mask = train_mask
graph_data.test_mask = test_mask



# Initialize model, criterion, and optimizer
model = TransactionGNN(num_node_features=node_features.shape[1], hidden_dim=128, num_classes=1).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
print("train_out shape:", train_out.shape)
print("train_labels shape:", train_labels.shape)
print("train_out values (sample):", train_out[:5])
print("train_labels values (sample):", train_labels[:5])

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index)
    train_out = out[graph_data.train_mask].squeeze()

    # Adjusting mask to target only nodes with labels
    # Assuming labels are for transaction nodes starting from 'transaction_node_start'
    adjusted_train_mask = train_mask[transaction_node_start:transaction_node_start + num_transactions]
    train_labels = graph_data.y[adjusted_train_mask].float()

    loss = criterion(train_out, train_labels)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Evaluation loop
model.eval()
with torch.no_grad():
    pred = model(graph_data.x, graph_data.edge_index)
    test_pred = pred[graph_data.test_mask].max(1)[1]
    test_labels = graph_data.y[graph_data.test_mask]
    test_pred, test_labels = test_pred.cpu(), test_labels.cpu()
    accuracy = accuracy_score(test_labels.numpy(), test_pred.numpy())
    precision = precision_score(test_labels.numpy(), test_pred.numpy())
    recall = recall_score(test_labels.numpy(), test_pred.numpy())
    f1 = f1_score(test_labels.numpy(), test_pred.numpy())
    conf_matrix = confusion_matrix(test_labels.numpy(), test_pred.numpy())
    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\nConfusion Matrix:\n{conf_matrix}")
