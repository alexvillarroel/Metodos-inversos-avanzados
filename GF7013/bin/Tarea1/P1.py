# -*- python -*-
# -*- coding: utf-8 -*-
"""
Francisco Hernán Ortega Culaciati
ortega.francisco@uchile.cl
Departamento de Geofísica - FCFM
Universidad de Chile


Make some testing of the multinomial distribution 

Modifications: 

"""
import numpy as np
import sys, os 
import matplotlib.pyplot as plt

# add GF7013 location to PYTHONPATH
path = os.path.join('..','..','..')
path = os.path.abspath(path)
sys.path.append(path)

from GF7013.probability_functions import pmf

# In order to instantiate a multinomial probability, I need three things:
# a) a random number generator instance
rng = np.random.default_rng(seed=42)

# b) a list of values representing a set of states for the random variable x.
values = np.array(['Spurious', 'Guess', 'No Idea', 'True value?', 'invalid?'])

# c) a list with the corresponding importance (unnormalized mass or probability) of 
#    each state described in values.
importance = np.array([1, 3, 2.2, 5, 0.1])

# assemble the dictionary of model parameters
par = {}
par['values'] = values
par['importance'] = importance
par['method'] = 'numpy'

# instantiate the multinomial pmf
multi = pmf.pmf_multinomial(par = par, rng = rng)

# print the probability of each discrete state
print('The probabilities for each discrete state are')
for i, value in enumerate(multi.values):
    print(value, 'with probability', multi.prob[i])
print('Total Probability is:', np.sum(multi.prob))

# draw some samples
Ns = 100_000
samples = multi.draw(Ns = Ns)

# make a histogram of the draws
count_values = np.zeros(multi.values.shape)
values2index = dict(zip(multi.values, range(len(multi.values))))

for value in samples:
    i = values2index[value]
    count_values[i] += 1

normalized_count_values = count_values / np.sum(count_values)

width = 0.25
N = len(multi.values)
index = np.arange(N)
prob = np.array([multi.eval(x) for x in multi.values])

# instantiate the second multinomial pmf
par2 = {}
par2['values'] = values
par2['importance'] = importance
par2['method'] = 'analog'

multi2 = pmf.pmf_multinomial(par = par2, rng = rng)
# print the probability of each discrete state

samples2 = multi2.draw(Ns = Ns)

# make a histogram of the draws
count_values2 = np.zeros(multi2.values.shape)
values2index2 = dict(zip(multi2.values, range(len(multi2.values))))
for value in samples2:
    i = values2index2[value]
    count_values2[i] += 1
normalized_count_values2 = count_values2 / np.sum(count_values2)

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.bar(index, normalized_count_values, width, color ='red', label = 'Normalized Sample Count')
ax.bar(index + width, prob, width, color = 'blue', label = 'Probability of each state' )
ax.bar(index + 2*width, normalized_count_values2, width, color ='green', label = 'Normalized Sample Count (analog)' )
ax.set_xticks(index + width/2, multi.values)
ax.set_title(f'Ns = {Ns:.1e} samples normalized count v/s probability')
ax.set_xlabel('Label of each discrete state')
ax.legend()
plt.show()
