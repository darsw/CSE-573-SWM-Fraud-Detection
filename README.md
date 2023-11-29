# CSE-573-SWM-Fraud-Detection
Fraud Detection With Graph Databases and Machine Learning

## Data
This project consists of the following datasets:
- **BankSim Dataset**: `bs140513_032310.csv`
- **Graph-Based Data**: `graph_features_dataset.csv`

## Code
### Python Scripts
- `graph_featuresset.py`: Generates graph features from Neo4j and exports to `graph_features_dataset.csv`.
- `graphdb_creation.cyp`: Contains commands for pushing data into the Neo4j tool.
- `graphplot.py`: Plots the graph using Matplotlib in Python.
- `graphfeatures.py`: Generates graph features using Python.

### Features Extracted
The following features are extracted:
- `merchDegree`
- `custDegree`
- `merchCloseness`
- `custCloseness`
- `custPageRank`
- `merchPageRank`
- `custBetweeness`
- `merchBetweeness`
- `merchlouvain`
- `custlouvain`
- `merchCommunity`
- `custCommunity`

### Web Files
- `index.html`: HTML file for data visualization.
- `main.js`: JavaScript code for data visualization.
