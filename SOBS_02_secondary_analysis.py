#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:22:29 2024

@author: richiercj
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import f_oneway, shapiro
import statsmodels.stats.multitest as mt
import seaborn as sns
import matplotlib.pyplot as plt



print('Running secondary analysis.')
print('')
# Import the dataframe
df = pd.read_csv('/Users/richiercj/Desktop/SOBS/data/SOBS_clustered.csv')

# Selection of input data type 
data_type = 'original' 
# data_type = 'z_scored'
# data_type = 'min_max'

cluster_variables = ['ASML', 'HUMN', 'NATN', 'OPMN'] 

if data_type == 'original':
    # cluster_variables = ['ASML', 'HUMN', 'NATN', 'OPMN', 'SoB'] # includes sense of belonging 
    cluster_variables = ['ASML', 'HUMN', 'NATN', 'OPMN'] 
    data = df.loc[:, cluster_variables]


# Define the column names for each group
cul_rac_columns = ['IRRS1', 'IRRS3', 'IRRS5', 'IRRS6', 'IRRS12', 'IRRS15', 'IRRS16', 'IRRS17', 'IRRS18', 'IRRS20']
ind_rac_columns = ['IRRS2', 'IRRS7', 'IRRS9', 'IRRS11', 'IRRS14', 'IRRS21']
inst_rac_columns = ['IRRS4', 'IRRS8', 'IRRS10', 'IRRS13', 'IRRS19', 'IRRS22']
# Sum across the specified columns and create new columns
df['IRRS_cul_rac'] = df[cul_rac_columns].sum(axis=1)
df['IRRS_ind_rac'] = df[ind_rac_columns].sum(axis=1)
df['IRRS_inst_rac'] = df[inst_rac_columns].sum(axis=1)

for col in data[:-1]:  # Assuming the last column is the cluster label
    plt.figure()
    sns.violinplot(x='cluster', y=col, data=df)
    plt.title(f'Distribution of {col} in each cluster')
    plt.show()



# Assuming `data` is your DataFrame and 'group' is your grouping variable
p_values = []
variables = ['IRRS_cul_rac', 'IRRS_ind_rac', 'IRRS_inst_rac', 'SoB']  # List your variables here
for var in variables:
    print('***************************')
    print(f'Running analysis for {var}')
    print('***************************')
    print(f'Mean and SD for {var}')
    print(np.mean(df[var]))
    print(np.std(df[var]))

    print('Generating plots...')
    print('')
    # Create histogram
    sns.histplot(df[var], bins=10, kde=False)  # kde for kernel density estimate
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {var}')
    plt.show()

    # Make violin plot 
    plt.figure()
    sns.violinplot(x='cluster', y=var, data=df)
    plt.title(f'Distribution of {var} in each cluster')
    plt.show()

    # Lilliefors test and levee's test

    print(f'Assumptions tests for {var}:')
    shaprio_result = shapiro(df[var])
    print(f"Shaprio Test for {var}: Statistic = {shaprio_result[0]}, p-value = {shaprio_result[1]}")
    if shaprio_result[1] <= .05:
        normality = False
        print(f'* WARNING! * Assumption of normality violated for {var}!')
    if shaprio_result[1] > .05:
        normality = True
        print(f'{var} is normally distributed.')
    
    # Levene's test
    levene_result = stats.levene(df[df['cluster'] == 0][var],
                                 df[df['cluster'] == 1][var],
                                 df[df['cluster'] == 2][var])
    print(f"Levene's Test: Statistic = {round(levene_result[0], 2)}, p-value = {levene_result[1]}")
    if levene_result[1] <= .05:
        equal_variances = False
        print(f'* WARNING! * Variances for all three clusters are not homegenous for {var}!')
    if levene_result[1] > .05:
        equal_variances = True
        print(f'Variances for {var} are homogenous.')
    
    print('')
    print('Running ANOVAS.')

    # Perform ANOVA for each variable
    stat, p = f_oneway(df[df['cluster'] == 0][var],
                       df[df['cluster'] == 1][var],
                       df[df['cluster'] == 2][var]
                       )
    p_values.append(p)
    print(f"Anova Test Statistic for {var}: {round(stat, 2)}")
    print('')
    

# Apply multiple testing correction, e.g., Bonferroni
print('********************************************************')
print(f'Corrected p values after running {len(variables)} tests:')
corrected_p = mt.multipletests(p_values, method='bonferroni')
print('********************************************************')
print('Corrected P-values:', corrected_p[1])
print('')
print('Running tukey tests...')
for var in variables:
    print('************************')
    print(f'Tukey test for {var}')
    print('************************')
    # Tukey's tests
    cluster_0 = df[df['cluster'] == 0][var]
    cluster_1 = df[df['cluster'] == 1][var]
    cluster_2 = df[df['cluster'] == 2][var]


    # Performing Tukey's HSD test
    tukey_result = stats.tukey_hsd(cluster_0, cluster_1, cluster_2)
    print(tukey_result)




