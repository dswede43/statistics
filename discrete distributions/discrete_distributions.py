#!/usr/bin/python3

#Discrete probability distributions
#---
#This script defines functions to create different discrete probability distributions.
#These include: binomial, multinomial, negative binomial, and poisson.

#import libraries
import numpy as np
import math
import matplotlib.pyplot as plt

#define global variables
dir = "/path/to/directory/"
rng = np.random.default_rng(123) #set seed
n = 10 #number of trials
p = 0.4 #probability of success
probs = [1/3, 1/3, 1/3] #probability of successes for multiple outcomes
r = 2 #number of successes
rate = 1 #expected number of successes over fixed interval of time
n_iter = 1000 #number of iterations


#Binomial distribution
#---
#function to generate a binomial distribution
def sample_from_binomial(n, p, n_sample = 1, random_state = None):
    if random_state == None:
        random_state = np.random.default_rng()
    
    if n_sample < 1:
        raise ValueError("Must have at least one sample")
        
    Y = []
    for _ in range(n_sample): #for each sample
        successes = 0
        
        #count the number of successes for each fixed number of independent Bernoulli trials
        for _ in range(n): #for each independent Bernoulli trial
            outcome = random_state.random() #generate a random number between 0 and 1
            if outcome < p:
                successes += 1 #add a successful trial
                
        Y.append(successes) #append the number of successes in the fixed number of independent Bernoulli trials
        
    return np.array(Y) if n_sample > 1 else Y[0]


#Multinomial distribution
#---
#function to generate a multinomial distribution
def sample_from_multinomial(n, probabilities, n_sample = 1, random_state = None):
    if random_state == None:
        random_state = np.random.default_rng()
    
    if n_sample < 1:
        raise ValueError("Must have at least one sample")    
    
    if not isinstance(probabilities, list):
        raise ValueError("Probabilities must be a list")
    
    n_outcomes = len(probabilities) #number of possible outcomes
    Y = np.zeros((n_sample, n_outcomes), dtype = int) #generate the matrix of results
    for i in range(n_sample):
        successes = np.zeros(n_outcomes, dtype = int) #array of successes
        
        for _ in range(n):
            outcome = np.random.rand() #generate a random number between 0 and 1
            
            #determine the category based on the probabilities    
            cumulative_prob = 0
            for j, prob in enumerate(probabilities):
                cumulative_prob += prob
                if outcome < cumulative_prob:
                    successes[j] += 1 #add a successful trial
                    break
        
        Y[i] = successes
    
    return np.array(Y) if n_sample > 1 else Y[0]


#Negative binomial distribution
#---
#function to generate a negative binomial distribution
def sample_from_nb(r, p, n_sample, random_state = None):
    if random_state == None:
        random_state = np.random.default_rng()
    
    Y = []
    for _ in range(n_sample): #for each independent Bernoulli trial
        failures = 0
        successes = 0
        
        #count the number of failures before the rth success
        while successes < r:
            outcome = random_state.random() #generate a random number between 0 and 1
            if outcome < p:
                successes += 1 #add a successful trial
            else:
                failures += 1 #add a failed trial
                
        Y.append(failures) #append the number of failures before the rth success has occurred
        
    return np.array(Y) if n_sample > 1 else Y[0]


#Poisson distribution
#---
def sample_from_poisson(rate, n_sample, random_state = None):
    if random_state == None:
        random_state = np.random.default_rng()
    
    if n_sample < 1:
        raise ValueError("Must have at least one sample")
    
    Y = []
    for _ in range(n_sample):
        outcome = np.random.rand() #generate a random number between 0 and 1
        
        #initialize counter and cumulative probability
        successes = 0
        cumulative_prob = np.exp(-rate)
        
        #iterate until cumulative probability exceeds the random number
        while outcome >= cumulative_prob:
            successes += 1 #add a successful trial
            cumulative_prob += np.exp(-rate) * (rate ** successes) / math.factorial(successes)
        
        Y.append(successes)
    
    return np.array(Y) if n_sample > 1 else Y[0]


#Visualize the distributions with stem plots
#---
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (10, 10))
fig.suptitle("Discrete random variable distributions")

#binomial
binomial = sample_from_binomial(n, p, n_iter, rng)
x = np.sum(np.arange(n + 1).reshape(-1, 1) == binomial, axis=1)
ax1.vlines(np.arange(n + 1), np.zeros_like(x), x)
ax1.plot(np.arange(n + 1), x, 'o')
ax1.set_title(f"Binomial ~ (n = {n}, p = {p})")
ax1.set_ylabel('Frequency')
ax1.set_xlabel('Random variable (Y)')

#multinomial
multinomial = sample_from_multinomial(n, probs, n_iter, rng)
probs = [round(prob, 2) for prob in probs]
multinomial = multinomial.reshape(-1)
x = np.sum(np.arange(n + 1).reshape(-1, 1) == multinomial, axis=1)
ax2.vlines(np.arange(n + 1), np.zeros_like(x), x)
ax2.plot(np.arange(n + 1), x, 'o')
ax2.set_title(f"Multinomial ~ (n = {n}, p = {probs})")
ax2.set_ylabel('Frequency')
ax2.set_xlabel('Random variable (Y)')

#negative binomial
negative_binomial = sample_from_nb(r, p, n_iter, random_state = rng)
x = np.sum(np.arange(30).reshape(-1, 1) == negative_binomial, axis=1)
ax3.vlines(np.arange(30), np.zeros_like(x), x)
ax3.plot(np.arange(30), x, 'o')
ax3.set_title(f"Negative Binomial ~ (r = {r}, p = {p})")
ax3.set_ylabel('Frequency')
ax3.set_xlabel('Random variable (Y)')

#poisson
poisson = sample_from_poisson(rate, n_iter, random_state = rng)
x = np.sum(np.arange(30).reshape(-1, 1) == poisson, axis=1)
ax4.vlines(np.arange(30), np.zeros_like(x), x)
ax4.plot(np.arange(30), x, 'o')
ax4.set_title(f"Poisson ~ (lambda = {rate})")
ax4.set_ylabel('Frequency')
ax4.set_xlabel('Random variable (Y)')

plt.tight_layout()
plt.show()

#save the plots as a pdf
fig.savefig(dir + '/discrete distributions/discrete_distributions.pdf', format = 'pdf')
