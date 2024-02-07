#!/usr/bin/python3

#Maximum Likelihood Estimation (MLE)-Linear regression
#---
#This script demonstrates Maximum Likelihood Estimation (MLE) to estimate the beta parameters
#(beta0 and beta1) for a simple linear regression model.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Generate data
#---
n = 50 #sample size
beta0 = 2 #y-int
beta1 = 3 #slope
x = np.random.uniform(0, 10, n) #explanatory variable
noise = np.random.normal(0, 2, n) #normally distributed random error
y = beta0 + beta1 * x + noise #predictor variable


#MLE estimation
#---
#function to calculate the log-likelihood of y and x given beta0 and beta1
def likelihood(beta0, beta1):
    #calculate the predicted value for y
    y_pred = beta0 + beta1 * x
    
    #calculate the log-likelihood of y and x given beta0 and beta1
    #assume the errors are normally distributed: N~(0,1)
    likelihoods = -n * np.log(2 * np.pi) / 2 - n * np.log(1) - np.sum((y - y_pred)**2) / 4
    
    return likelihoods

#define the ranges for beta0 and beta1
beta0_range = np.linspace(0, 4, 100)
beta1_range = np.linspace(0, 4, 100)

#estimate the beta parameters using MLE
beta_likelihoods = []
for beta0 in beta0_range:
    for beta1 in beta1_range:
        beta_likelihoods.append({'beta0': beta0, 'beta1': beta1, 'likelihood': likelihood(beta0, beta1)})

#create a data frame of results
beta_likelihoods_df = pd.DataFrame(beta_likelihoods)

#find the estimates for beta0 and beta1 based on max log-likelihood
betas_mle = beta_likelihoods_df.loc[beta_likelihoods_df['likelihood'].idxmax()]
beta0_mle = betas_mle['beta0']
beta1_mle = betas_mle['beta1']

#print results
print(f"The MLE estimate for beta0 is {beta0_mle:.2f}.")
print(f"The MLE estimate for beta1 is {beta1_mle:.2f}.")


#Visualize MLE results
#---
#filter the data frame for beta0 and beta1 MLE estimates
beta0_df = beta_likelihoods_df[beta_likelihoods_df['beta1'] == beta1_mle]
beta1_df = beta_likelihoods_df[beta_likelihoods_df['beta0'] == beta0_mle]

#plot the results
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1, figsize = (8, 10))
fig.suptitle("MLE estimates of beta0 and beta1 for a simple linear regression")

#create beta0 subplot
ax1.plot(beta0_df['beta0'], beta0_df['likelihood'], linestyle = '-', color = 'green')
ax1.set_xlabel('Potential values of beta0')
ax1.set_ylabel('Likelihood')
ax1.axvline(x = beta0_mle, linestyle = '--', color = 'red')

#create beta1 subplot
ax2.plot(beta1_df['beta1'], beta1_df['likelihood'], linestyle = '-', color = 'blue')
ax2.set_xlabel('Potential values of beta1')
ax2.set_ylabel('Likelihood')
ax2.axvline(x = beta1_mle, linestyle = '--', color = 'red')

#linear regression subplot
x_line = np.linspace(0, 10, 10)
y_line = beta0_mle + beta1_mle * x_line
ax3.scatter(x, y)
ax3.plot(x_line, y_line, color = 'red', linestyle = '-')
ax3.set_xlabel('explanatory variable (x)')
ax3.set_ylabel('predictory variable (y)')

plt.tight_layout()
plt.show()

fig.savefig('MLE_linear_regression.pdf', format = 'pdf') #save the plot as a pdf
