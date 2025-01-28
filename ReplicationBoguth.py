#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 08:49:53 2025

@author: daniele bianchi
d.bianchi@qmul.ac.uk

"""

import pandas as pd
from helper_functions import *
import matplotlib.pyplot as plt


# Upload consumption data
sheet_name = 'Consumption_Data'
data = pd.read_excel('BoguthKuehn2013JF_Data.xlsx',sheet_name)
data.set_index('date',inplace=True)                     
                     
# Calculate total consumption and service share
data['TotConsumption']     = data['Non-durable'] + data['Service'] 
data['ServiceShare']       = data['Service']/data['TotConsumption']

# Calculate log differences
data['Delta Service']      = data['Service'].pipe(np.log).diff()
data['Delta ServiceShare'] = data['ServiceShare'].pipe(np.log).diff()

# Drop rows with NaN values
data = data.dropna()

# Define the model

model = JointRegimeSwitchingModel(data['Delta Service'], data['Delta ServiceShare'], debug=True)

# Try fitting with different initial values for robustness
initial_guesses = [
    np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 2.0, 2.0, 2.0, 2.0]),
    np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    np.array([-0.2, 0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2, 3.0, 3.0, 3.0, 3.0])
]

for i, guess in enumerate(initial_guesses):
    print(f"\nTrying initial guess set {i+1}")
    results = model.fit(guess)
    if results is not None and results['parameters'] is not None:
        print("Convergence achieved!")
        break

# Extract the parameters estimates and saves them in Excel
save_estimates_to_excel(results, 'parameters estimates.xlsx')

# Plot the high-state probabilities
# Extract filtered probabilities
filtered_probs = results['filtered_probabilities']

# Create DataFrame with meaningful column names
prob_df = pd.DataFrame(filtered_probs, 
                      columns=['P(Low μ, Low σ)', 
                              'P(Low μ, High σ)', 
                              'P(High μ, Low σ)', 
                              'P(High μ, High σ)'])

# If you have dates in your original data, you can add them as index
# Assuming your data has a DateIndex and same length as filtered probabilities
prob_df.index = data.index

# Calculate derived probabilities
prob_df['P(High Mean)'] = prob_df['P(High μ, Low σ)'] + prob_df['P(High μ, High σ)']
prob_df['P(High Vol)'] = prob_df['P(Low μ, High σ)'] + prob_df['P(High μ, High σ)']


# Extract the probabilities from the original paper
sheet_name = 'Sheet1'
beliefs_original = pd.read_excel('BoguthKuehn2013JF_Beliefs.xlsx',sheet_name)
beliefs_original.set_index('date',inplace=True)                     


# Plot the probabilities
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
   
# Plot high mean state probability
ax1.plot(prob_df['P(High Mean)'], 'b-', lw=1.5)
ax1.plot(beliefs_original['mu'], 'b--', lw=2)
ax1.set_title('Prior beliefs of High Mean State')
ax1.set_ylabel('Probability')
ax1.grid(True)
ax1.set_ylim(0, 1)
ax1.legend(['replicated','original'])
   
# Plot high volatility state probability
ax2.plot(prob_df['P(High Vol)'], 'r-', lw=1.5)
ax2.plot(beliefs_original['sig'], 'r--', lw=2)
ax2.set_title('Prior beliefs of High Volatility State')
ax2.set_xlabel('Time')
ax2.set_ylabel('Probability')
ax2.legend(['replicated','original'])
ax2.grid(True)
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('Prior beliefs replication.png', dpi=300, bbox_inches='tight')
plt.show()



