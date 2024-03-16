#!/usr/bin/python3

#Maximum Likelihood Estimation (MLE)-Normal Distribution
#---
#This script demonstrates Maximum Likelihood Estimation (MLE) to estimate the parameters for a normal distribution.
#ie. mean (mu) and standard deviation (sigma).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#define global variables
n = 100 #sample size

mu = 5 #population mean
upper_mu = 2 * mu #upper limit of potential mu's
lower_mu = -mu #lower limit of potential mu's

sigma = 2 #population standard deviation
upper_sigma = 2 * sigma #upper limit of potential sigma's
lower_sigma = 0.01 #lower limit of potential sigma's

#simulate a normal distribution
x = np.random.normal(mu, sigma, n)


#Estimate both the mean (mu) and standard deviation (sigma) for a normal distribution
#---
muhats = np.linspace(lower_mu, upper_mu, n)  # Potential estimates for mu
sigmahats = np.linspace(lower_sigma, upper_sigma, n)  # Potential estimates for sigma

#create a grid of mu and sigma combinations
mu_sigma_likelihoods = []

for muhat in muhats:
    for sigmahat in sigmahats:
        #calculate the log-likelihood of the data given mu and sigma
        sigma_likelihood = -n * np.log(2 * np.pi) / 2 - n * np.log(sigmahat) - np.sum((x - muhat) ** 2) / (2 * sigmahat ** 2)
        mu_sigma_likelihoods.append({'muhat': muhat, 'sigmahat': sigmahat, 'likelihood': sigma_likelihood})

#create a data frame of results
mu_sigma_df = pd.DataFrame(mu_sigma_likelihoods)

#find the estimates for mu and sigma based on max likelihood
mu_sigma_mle = mu_sigma_df.loc[mu_sigma_df['likelihood'].idxmax()]
mu_mle = mu_sigma_mle['muhat']
sigma_mle = mu_sigma_mle['sigmahat']

#print results
print(f"The MLE estimate for the mean (mu) is {mu_mle:.2f}.")
print(f"The MLE estimate for the standard deviation (sigma) is {sigma_mle:.2f}.")


#Visualize MLE results
#---

#filter the data frame for mu and sigma MLE estimates
mu_df = mu_sigma_df[mu_sigma_df['sigmahat'] == sigma_mle]
sigma_df = mu_sigma_df[mu_sigma_df['muhat'] == mu_mle]

#plot the results
fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 8))
fig.suptitle("MLE estimates of the mean (mu) and standard deviation (sigma) for a normal distribution")

#create mu subplot
ax1.plot(mu_df['muhat'], mu_df['likelihood'], linestyle = '-', color = 'green')
ax1.set_xlabel('Potential values of mu')
ax1.set_ylabel('Log-likelihood')
ax1.axvline(x = mu_mle, linestyle = '--', color = 'red')

#create sigma subplot
ax2.plot(sigma_df['sigmahat'], sigma_df['likelihood'], linestyle = '-', color = 'blue')
ax2.set_xlabel('Potential values of sigma')
ax2.set_ylabel('Log-likelihood')
ax2.axvline(x = sigma_mle, linestyle = '--', color = 'red')

plt.show()

fig.savefig('MLE_normal_distribution.pdf', format = 'pdf') #save the plot as a pdf
