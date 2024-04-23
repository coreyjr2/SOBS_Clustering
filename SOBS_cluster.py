# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Import the dataframe
df = pd.read_csv('/Users/richiercj/Downloads/SOBS_python.csv')

# store the total list of variables
variables = list(df.columns)

# Variables to cluster
cluster_variables = ['ASML', 'HUMN', 'NATN', 'OPMN']

df_to_cluster = df.loc[:, cluster_variables]

# Perform hierarchical clustering
linkage_matrix = sch.linkage(df_to_cluster, method='ward')

# Plot dendrogram with color-coded branches
plt.figure(figsize=(12, 6))
dendrogram = sch.dendrogram(linkage_matrix, color_threshold=10)  # Adjust color_threshold as needed
plt.title('Dendrogram with Color-Coded Branches')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distances')
plt.show()

'''
It appears that after
looking at a graphical representation of the dendrogram, 
that three clusters appear to be the best solution.
'''

# Extract clusters using a chosen color threshold
color_threshold = 10  # Adjust as needed
clusters = sch.fcluster(linkage_matrix, t=color_threshold, criterion='distance')
np.unique(clusters)

df['HC_Cluster_assignment'] = clusters

# Group by 'grouping_column' and calculate descriptive statistics
grouped_stats = df.groupby('HC_Cluster_assignment')[cluster_variables].describe()

# Print the result
print(grouped_stats)
