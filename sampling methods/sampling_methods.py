#!/usr/bin/env python3

#Sampling methods: simple, stratified, and cluster (sampling without replacement)
#---
#This script simulates simple, stratified, and cluster sampling methods to estimate the population mean of a variable.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#define global variables
num_var = 'dep_delay' #numeric variable
strat_var = 'origin' #stratification variable
clust_var = 'carrier' #cluster variable
n_iter = 1000 #number of sampling iterations
n = 5000 #sample size
n_clust = 3 #define the number of clusters to sample

#load the data
#This data contains information about all flights that departed from NYC to destinations in the US, Puerto Rico
#and American Virgin Islands in 2013. Data obtained from https://github.com/tidyverse/nycflights13?tab=readme-ov-file
#The variable of interest is the flight departure delays in minutes.
dir = "/path/to/directory/"
df = pd.read_csv(dir + "data/nycflights.csv")
cols_to_keep = ['origin','carrier','dep_delay']
df = df[cols_to_keep]
df = df.dropna()


#Simple random sample (SRS)
#---
print(f"Running {n_iter} iterations of simple random sampling...")

rng = np.random.default_rng(123) #set the seed
N = len(df) #define the population size
srs_means = np.zeros(n_iter)
srs_mean_sds = np.zeros(n_iter)

for i in range(n_iter):
    sample = df.sample(n, random_state = rng) #take a simple random sample
    srs_means[i] = sample[num_var].mean() #estimate the sample mean
    srs_var = sample[num_var].var() #estimate the sample variance
    srs_mean_sds[i] = np.sqrt(((srs_var / n) * (1 - (n / N)))) #estimate SD of the sample mean


#Stratified sampling (proportional allocation)
#---
print(f"Running {n_iter} iterations of stratified sampling using proportional allocation...")

N = len(df) #define population size
p = n / N #calculate proportion for proportional allocation

rng = np.random.default_rng(123) #set the seed
strats = df[strat_var].unique() #create an array with all stratum names
strat_means = np.zeros(n_iter)
strat_mean_sds = np.zeros(n_iter)

for i in range(n_iter):
    strat_estimates = {} #define an empty dictionary
    for strat in strats:
        strat_df = df[df[strat_var] == strat] #filter rows for the stratum
        nh = int(round(p * len(strat_df), 0)) #stratum sample size
        Nh = len(strat_df) #stratum population size
        sample = strat_df.sample(nh , random_state = rng) #simple random sample of the stratum
        mean_h = sample[num_var].mean() #mean of stratum
        var_h = sample[num_var].var() #variance of stratum
        strat_estimates[strat] = [mean_h, var_h, nh, Nh] #store stratum-level estimates in the dictionary
        strat_estimates_df = pd.DataFrame(strat_estimates).T #convert the dictionary to a data frame
        strat_estimates_df.columns = ['mean_h','var_h','nh','Nh'] #add column headers
    mean_h = strat_estimates_df.mean_h
    var_h = strat_estimates_df.var_h
    nh = strat_estimates_df.nh
    Nh = strat_estimates_df.Nh
    strat_means[i] = sum((Nh / N) * mean_h) #estimate of population mean
    strat_mean_sds[i] = np.sqrt(sum((1 - (nh / Nh)) * (Nh / N)**2 * (var_h / nh))) #estimate standard error of mean estimate


#Cluster sampling (one-stage: unequal cluster sizes)
#---
print(f"Running {n_iter} iterations of one-stage cluster sampling...")

np.random.seed(123) #set the seed
clusters = df[clust_var].unique() #find all unique clusters
N = len(clusters) #total number of clusters
clust_means = np.zeros(n_iter)
clust_mean_sds = np.zeros(n_iter)

for i in range(n_iter):
    sample_clusters = np.random.choice(clusters, size = n_clust, replace = False) #take a random sample of clusters
    clust_estimates = {}
    for sample_cluster in sample_clusters:
        clust_df = df[df[clust_var] == sample_cluster] #filter rows for the sampled cluster
        Mi = len(clust_df) #cluster size
        ti = clust_df[num_var].sum() #cluster total
        ybari = clust_df[num_var].mean() #cluster mean
        clust_estimates[sample_cluster] = [ti, ybari, Mi] #store cluster-level estimates
        clust_estimates_df = pd.DataFrame(clust_estimates).T #convert to data frame
        clust_estimates_df.columns = ['ti', 'ybari','Mi'] #add column headers
    ti = clust_estimates_df.ti #cluster totals
    ybari = clust_estimates_df.ybari #cluster means
    Mi = clust_estimates_df.Mi #cluster sizes
    clust_mean = sum(ti) / sum(Mi) #estimate population mean
    clust_means[i] = clust_mean
    vt = np.sum((Mi * ybari - Mi * clust_mean)**2) / (n - 1) #cluster total variance
    clust_mean_sds[i] = np.sqrt((1 - n_clust / N) * (N / Mi.sum())**2 * (vt / n_clust)) #estimate standard error of mean estimate


#Visualize all sampling results
#---
#population mean estimates
mean_results = pd.DataFrame({'SRS': srs_means, 'STRAT': strat_means, 'CLUST': clust_means}) #estimated population means

#standard error distributions
SE_results = pd.DataFrame({'SRS': srs_mean_sds, 'STRAT': strat_mean_sds, 'CLUST': clust_mean_sds}) #estimated standard errors of mean

#define subplots
fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 8))
fig.suptitle("Simple, stratified, and cluster sampling estimates\nfor the population mean")

#create the first subplot
mins = []
maxs = []
for i in mean_results.columns: #define the bin widths for the histograms
    mins.append(min(mean_results[i]))
    maxs.append(max(mean_results[i]))

bins = np.linspace(min(mins), max(maxs), num = 50)
ax1.hist(mean_results['SRS'], bins = bins, alpha = 1, label = 'SRS')
ax1.hist(mean_results['STRAT'], bins = bins, alpha = 0.5, label = 'STRAT')
ax1.hist(mean_results['CLUST'], bins = bins, alpha = 0.5, label = 'CLUST')
ax1.set_ylabel("Frequency")
ax1.set_xlabel("Distribution of mean estimates")
ax1.legend()

#create the second subplot
mins = []
maxs = []
for i in SE_results.columns: #define the bin widths for the histograms
    mins.append(min(SE_results[i]))
    maxs.append(max(SE_results[i]))

bins = np.linspace(min(mins), max(maxs), num = 50)
ax2.hist(SE_results['SRS'], bins = bins, alpha = 1, label = 'SRS')
ax2.hist(SE_results['STRAT'], bins = bins, alpha = 0.5, label = 'STRAT')
ax2.hist(SE_results['CLUST'], bins = bins, alpha = 0.5, label = 'CLUST')
ax2.set_ylabel("Frequency")
ax2.set_xlabel("Distribution of standard errors")
ax2.legend()

plt.show()

fig.savefig(dir + 'sampling methods/sampling_methods.pdf', format = 'pdf') #save the plot as a pdf
