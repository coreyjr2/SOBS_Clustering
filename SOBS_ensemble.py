# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 17:14:06 2023

@author: Corey Richier
"""

####################
# Import libraries #
#################### 

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns

############################
# Data Import and handling #
############################

# Some variables for the script
handle_outliers = False

# Import the dataframe
df = pd.read_csv('/Users/richiercj/Desktop/SOBS/data/SOBS_python.csv')

# Selection of input data type 
data_type = 'original' 
# data_type = 'z_scored'
# data_type = 'min_max'

if data_type == 'original':
    # cluster_variables = ['ASML', 'HUMN', 'NATN', 'OPMN', 'SoB'] # includes sense of belonging 
    cluster_variables = ['ASML', 'HUMN', 'NATN', 'OPMN'] 
    data = df.loc[:, cluster_variables]
    
elif data_type == 'z_scored':
    z_score_cluster_variables = ['Z_ASML', 'Z_HUMN', 'Z_NATN', 'Z_OPMN']
    data = df.loc[:, z_score_cluster_variables]
    
elif data_type == 'min_max':
    min_max_scaler = MinMaxScaler()
    cluster_variables = ['ASML', 'HUMN', 'NATN', 'OPMN']
    data = pd.DataFrame(min_max_scaler.fit_transform(df.loc[:, cluster_variables]))

#####################
# Data Descriptives #
#####################

# print descriptes for each data
data.describe()

# Print hisograms for the data
for column in data.columns:
    plt.figure(figsize=(10, 4))  # Set the figure size
    df[column].hist(bins=30)  # Adjust the number of bins for better resolution
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Optional outlier handling 
if handle_outliers:
    outlier_indices = {}
    for col in data.select_dtypes(include=[np.number]).columns:  # Ensure the column is numeric
        mean = data[col].mean()
        std = data[col].std()
        # Identify outliers
        outliers = data[(data[col] < (mean - 3 * std)) | (data[col] > (mean + 3 * std))]
        outlier_indices[col] = outliers.index.tolist()
        print(f"Outliers in {col}: {outlier_indices[col]}")
    
    # Create a boolean mask with all True values
    mask = pd.Series([True] * len(data))
    
    # Iterate over the dictionary to update the mask
    for col, indices in outlier_indices.items():
        mask.loc[indices] = False
    
    # Now, 'mask' is True for rows that are not outliers
    data = data[mask]

# correlate all the variables with one another
correlation_matrix = data.corr() 
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# visualize data reduced in PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data)

plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1]) 
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA - 2D Representation of the Data')
plt.show()

######################################
# Determining the Number of Clusters #
######################################

# Initializing lists to store metrics for each clustering method
# Summary lists
results_dict = {}

# Loop over a range of 2-10 clusters to establish fit metrics 
K = range(2, 11)  # Range for number of clusters
for k in K:
    
    # Initialize a sub-dictionary for this number of clusters
    results_dict[k] = {}
    
    # KMeans
    kmeans = KMeans(n_clusters = k, n_init = 100)
    kmeans.fit(data)
    results_dict[k]['kmeans_inertia'] = kmeans.inertia_
    results_dict[k]['kmeans_silhouette'] = silhouette_score(data, kmeans.labels_)
    results_dict[k]['kmeans_calinski_harabasz'] = calinski_harabasz_score(data, kmeans.labels_)
    results_dict[k]['kmeans_davies_bouldin'] = davies_bouldin_score(data, kmeans.labels_)
    results_dict[k]['kmeans_labels'] = kmeans.labels_  

    # Agglomerative Clustering
    agglo = AgglomerativeClustering(n_clusters = k)
    agglo_labels = agglo.fit_predict(data)
    results_dict[k]['agglo_silhouette'] = silhouette_score(data, agglo_labels)
    results_dict[k]['agglo_calinski_harabasz'] = calinski_harabasz_score(data, agglo_labels)
    results_dict[k]['agglo_davies_bouldin'] = davies_bouldin_score(data, agglo_labels)
    results_dict[k]['agglo_labels'] = agglo_labels

    # Gaussian Mixture Models
    gmm = GaussianMixture(n_components = k)  
    gmm_labels = gmm.fit_predict(data)
    results_dict[k]['gmm_silhouette'] = silhouette_score(data, gmm_labels)
    results_dict[k]['gmm_calinski_harabasz'] = calinski_harabasz_score(data, gmm_labels)
    results_dict[k]['gmm_davies_bouldin'] = davies_bouldin_score(data, gmm_labels)
    results_dict[k]['gmm_labels'] = gmm_labels
    
    # Spectral Clustering
    spectral = SpectralClustering(n_clusters=k, affinity='nearest_neighbors')
    spectral_labels = spectral.fit_predict(data)
    results_dict[k]['spectral_silhouette'] = silhouette_score(data, spectral_labels)
    results_dict[k]['spectral_calinski_harabasz'] = calinski_harabasz_score(data, spectral_labels)
    results_dict[k]['spectral_davies_bouldin'] = davies_bouldin_score(data, spectral_labels)
    results_dict[k]['spectral_labels'] = spectral_labels
    
# Plotting metrics
plt.figure(figsize=(20, 20), dpi=150)  # Adjusted figure size for clarity

# Create a list of metrics for convenience
metrics = ['inertia', 'silhouette', 'calinski_harabasz', 'davies_bouldin']

plt.figure(figsize=(12, 8))  # Adjust the size as needed
for i, metric in enumerate(metrics):
    plt.subplot(len(metrics), 1, 1 + i)
    plt.plot(K, [results_dict[k]['kmeans_' + metric] for k in K], 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel(metric.capitalize())
    plt.title('KMeans: ' + metric.capitalize())
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))  # Adjust the size as needed
for i, metric in enumerate(metrics):
    if metric == 'inertia': continue
    plt.subplot(len(metrics) - 1, 1, i)  # Adjust the index if needed
    plt.plot(K, [results_dict[k]['agglo_' + metric] for k in K], 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel(metric.capitalize())
    plt.title('Agglomerative: ' + metric.capitalize())
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))  # Adjust the size as needed
for i, metric in enumerate(metrics):
    if metric == 'inertia': continue
    plt.subplot(len(metrics) - 1, 1, i)  # Adjust the index if needed
    plt.plot(K, [results_dict[k]['gmm_' + metric] for k in K], 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel(metric.capitalize())
    plt.title('GMM: ' + metric.capitalize())
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))  # Adjust the size as needed
for i, metric in enumerate(metrics):
    if metric == 'inertia': continue
    plt.subplot(len(metrics) - 1, 1, i)  # Adjust the index if needed
    plt.plot(K, [results_dict[k]['spectral_' + metric] for k in K], 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel(metric.capitalize())
    plt.title('Spectral Clustering: ' + metric.capitalize())
plt.tight_layout()
plt.show()

# a dictionary without the labels to find the min and max
inertia_list = []
silhouette_list = []
calinski_harabasz_list = []
davies_bouldin_list = []
interpreted_results = {}
for k in K:
    interpreted_results[k] = {}
    for metric_name, value in results_dict[k].items():
        # Skip if the metric is labels
        if 'labels' in metric_name:
            continue        
        # Higher is better metrics
        if 'silhouette' in metric_name: 
            silhouette_list.append(value)
        if 'calinski_harabasz' in metric_name:
            calinski_harabasz_list.append(value)
        # Lower is better metrics
        if 'inertia' in metric_name:
            inertia_list.append(value)
        if 'davies_bouldin' in metric_name:
            davies_bouldin_list.append(value)
        interpreted_results[k][metric_name] = value

min_max_values = {
    'silhouette': {'min': min(silhouette_list), 'max': max(silhouette_list)},
    'calinski_harabasz': {'min': min(calinski_harabasz_list), 'max': max(calinski_harabasz_list)},
    'inertia': {'min': min(inertia_list), 'max': max(inertia_list)},
    'davies_bouldin': {'min': min(davies_bouldin_list), 'max': max(davies_bouldin_list)}
}
     
for k in K:
    combined_metric = 0
    for metric_name, value in interpreted_results[k].items():

        if'silhouette' in metric_name:
            # Normalize (higher is better)
            normalized_value = (value - min_max_values['silhouette']['min']) / (min_max_values['silhouette']['max'] - min_max_values['silhouette']['min'])
        if'calinski_harabasz' in metric_name:
            # Normalize (higher is better)
            normalized_value = (value - min_max_values['calinski_harabasz']['min']) / (min_max_values['calinski_harabasz']['max'] - min_max_values['calinski_harabasz' ]['min'])
        if 'inertia' in metric_name:
            normalized_value = (value - min_max_values['inertia']['max']) / (min_max_values['inertia']['min'] - min_max_values['inertia']['max'])
        if 'davies_bouldin' in metric_name:
            # Normalize inverted value (lower is better)
            normalized_value = (value - min_max_values['davies_bouldin']['max']) / (min_max_values['davies_bouldin']['min'] - min_max_values['davies_bouldin']['max'])
        combined_metric += normalized_value

    # Store the average combined metric
    interpreted_results[k]['combined_metric'] = combined_metric / 4

# Store the best number of clusters based on the global optimum of each of the four metrics
best_k = max(interpreted_results, key=lambda x: interpreted_results[x]['combined_metric'])
print("Best k based on combined metric:", best_k)

###############################################
# Generating the predictions for each cluster #
###############################################

# Utilize the number of clusters found to be the best 
num_clusters = best_k
print(f'Running cluster ensemble with {best_k} clusters.')

# Perform clustering with each method
kmeans_labels = KMeans(n_clusters=num_clusters, n_init = 100).fit_predict(data)
agglo_labels = AgglomerativeClustering(n_clusters=num_clusters).fit_predict(data)
gmm_labels = GaussianMixture(n_components=num_clusters).fit_predict(data)
spectral_labels = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors').fit_predict(data)

n_samples = data.shape[0]
co_association_matrix = np.zeros((n_samples, n_samples))


# Gneration of distance matrix based on the labels estimated by each cluster method
for labels in [kmeans_labels, agglo_labels, gmm_labels, spectral_labels]:
    for i in range(n_samples):
        for j in range(n_samples):
            if labels[i] == labels[j]:
                co_association_matrix[i, j] += 1

# Normalize the co-association matrix
co_association_matrix /= 4  # Since there are 4 clustering methods

# Fit the model with the co-association matrix
# Initialize Agglomerative Clustering with precomputed affinity
agg_clustering = AgglomerativeClustering(n_clusters=num_clusters, 
                                         metric='precomputed', 
                                         linkage='average')  # 'average', 'complete', 'single', etc.

consensus_labels = agg_clustering.fit_predict(1 - co_association_matrix)  # 1 - matrix, if the matrix is similarity
print("Consensus for each subject's cluster assignment determined.")

# Plot the clusters in a PCA graph
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=consensus_labels, cmap='viridis', alpha=0.5)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Cluster Assignment Visualization')
plt.show()

###########################################
# Calculate descriptives for each cluster #
###########################################

# Add the cluster ID to the original dataframe
df['Cluster'] = consensus_labels

# Select demographic variables
demographic_variables = ['Cluster', 'Gender', 'Age', 'Marital', 'Num_of_Children', 'Attending_4year',
                         'Highest_Education', 'Mom_Education', 'Dad_Education', 'Employment',
                         'Gross', 'Religion', 'Religiosity', 'Racial_Comp_Comm', 'Racial_Comp_Peers' , 
                         'ASML', 'HUMN', 'NATN', 'OPMN',
                         'SoB']
                         

# Print the cluster group membership totals 
df['Cluster'].value_counts()

# Group the dataframe by each cluster
grouped_df = df[demographic_variables].groupby('Cluster')

# To get mean values for each column in each cluster
mean_values = grouped_df.mean()

# To get median values
median_values = grouped_df.median()

# To get standard deviation
std_dev_values = grouped_df.std()

print("Mean values per cluster:\n", mean_values)
print("\nMedian values per cluster:\n", median_values)
print("\nStandard deviation per cluster:\n", std_dev_values)


# Save the cluster labels
df["cluster"] = consensus_labels
df.to_csv('/Users/richiercj/Desktop/SOBS/data/SOBS_clustered.csv')













