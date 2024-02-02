#!/usr/bin/python3

#Bootstrapping
#---
#This script demonstrates non-parametric bootstrapping to estimate the population mean of a random variable.

#load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#define global variables
n_SRS = 20 #size of simple random sample
n_boot = 1000 #bootstrap sample size
n_iter = 1000 #number of bootstrap iterations
rng = np.random.default_rng(123) #set the seed


#load the data
#This data contains information about all flights that departed from NYC to destinations in the US, Puerto Rico
#and American Virgin Islands in 2013. Data obtained from https://github.com/tidyverse/nycflights13?tab=readme-ov-file
#The variable of interest is the flight departure delays in minutes.
df = pd.read_csv("./nycflights.csv")
df = df['dep_delay']
df = df.dropna()
sample_df = df.sample(n_SRS, random_state = rng, replace = False) #take a simple random sample


#Complete bootstrapping
#---
boot_means = np.zeros(n_iter)

for i in range(n_iter):
    sample_boot = sample_df.sample(n_boot, random_state = rng, replace = True) #sample with replacement
    boot_means[i] = sample_boot.mean() #calculate the bootstrap sample mean

#obtain the 95% confidence intervals
lower_CI = np.percentile(boot_means, 2.5)
upper_CI = np.percentile(boot_means, 97.5)

print(f"The 95% confidence interval for the bootstrapped population estimate of the mean is between {lower_CI:.2f} and {upper_CI:.2f}. The actual population mean is {df.mean():.2f}.")


#Visualize the results
#---
fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 8))
fig.suptitle("Bootstrapped estimates for the mean of a random variable")

#create sample distribution
ax1.hist(sample_df, bins = 30, color = 'blue')
ax1.set_xlabel('Sample of flight departure delays (min) (n = 20)')
ax1.set_ylabel('Frequency')

#create bootstrap distribution
ax2.hist(boot_means, bins = 30, color = 'green')
ax2.set_xlabel('Bootstrapped estimates of the population mean')
ax2.set_ylabel('Frequency')
ax2.axvline(x = lower_CI, linestyle = '--', color = 'red')
ax2.axvline(x = upper_CI, linestyle = '--', color = 'red')

plt.show()

#save the plots as a pdf
fig.savefig('boostrapping_random_variable.pdf', format = 'pdf')
