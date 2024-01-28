#!/usr/bin/env python3

#Central Limit Theorem
#---
#This script demonstrates the result of central limit theorem using simple random sampling (SRS).
#Given a large enough samples size, the sampling distribution of the mean converges to a standard normal
#distribution, regardless of the population distribution (normal, poisson, binomial, exponential).

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load data
#This data contains information about all flights that departed from NYC () to destinations in the US, Puerto Rico
#and American Virgin Islands in 2013. Data obtained from https://github.com/tidyverse/nycflights13?tab=readme-ov-file
#The variable of interest is the flight departure delays in minutes.
df = pd.read_csv("./nycflights.csv")
cols_to_keep = ['dep_delay']
df = df[cols_to_keep]

#visualize the population distribution of flight departure delays
dep_delay = df['dep_delay']
plt.hist(dep_delay, bins = 50)
plt.title("Distribution of flight departure delays (min)")
plt.savefig('population_distribution.pdf', format = 'pdf') #save the plot as a pdf
plt.show()

#simple random sampling
rng = np.random.default_rng(123) #set the seed

n_samples = [10,25,50,100] #define sample sizes
n_iter = 1000 #define the number of iterations
N = len(df) #define the population size

srs_means_dict = {} #create an empty dictionary

for i in n_samples:
    srs_means = np.zeros(n_iter) #create an array of zeros

    for j in range(n_iter):
        sample = df['dep_delay'].sample(i, random_state = rng, replace = False) #take an SRS
        srs_means[j] = sample.mean() #estimate sample mean

    srs_means_dict[i] = srs_means #store the results in the dictionary

#convert the dictionary into a data frame
srs_means_df = pd.DataFrame(srs_means_dict)
srs_means_df.columns = ['n=10','n=25','n=50','n=100']

#visualize the SRS distributions
mins = []
maxs = []
for i in srs_means_df.columns:
    mins.append(min(srs_means_df[i]))
    maxs.append(max(srs_means_df[i]))

#define the bin widths for the histograms
bins = np.linspace(min(mins), max(maxs), num = 50)

fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (10, 6), sharex = True, sharey = True) #create four subplots
fig.suptitle("Distribution of mean flight departure delays (min) for simple random sampling") #create subplot title
axes = axes.flatten() #flatten the axes for easier iteration

for i, col in enumerate(srs_means_df.columns):
    axes[i].hist(srs_means_df[col], bins = bins)
    axes[i].set_title(col)
    axes[i].set_xlabel('Departure delay (min)')
    axes[i].set_ylabel('counts')

fig.savefig('SRS_distributions.pdf', format = 'pdf') #save the plot as a pdf

#The sampling distributions demonstrate the convergence to a standard normal distribution of sampling means
#as the sample size increases.