# CSE-573-SWM-Fraud-Detection
Fraud Detection With Graph Databases and Machine Learning

# Data
Consist of Dataset BankSim Dataset : bs140513_032310.csv 
Graph based Data : graph_features_dataset.csv

# Code
  
graph_featuresset.py- Generates graph features from neo4j to graph_features_dataset.csv
graphdb_creation.cyp - Commands for pushing the data into neo4j tool.
Features Extracted : merchDegree ,custDegree , merchCloseness , custCloseness , custPageRank , merchPageRank , custBetweeness ,merchBetweeness ,merchlouvain , custlouvain ,merchCommunity ,custCommunit
graphplot.py - Plotting the graph using matplot lib in python
graphfeatures.py - Generating graph features using python

index.html - HTML file for data visualisation
main.js - Javascript code data visualisation
