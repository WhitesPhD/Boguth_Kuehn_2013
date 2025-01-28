#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 08:48:37 2025

@author: daniele bianchi
d.bianchi@qmul.ac.uk
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import logsumexp
import matplotlib.pyplot as plt

from tqdm import tqdm
import warnings
import pandas as pd


class JointRegimeSwitchingModel:
    def __init__(self, x1, x2, debug=False):
        self.debug = debug
        self.x1 = x1
        self.x2 = x2
        
        # Calculate initial guesses based on data
        self.x1_mean, self.x1_std = np.mean(x1), np.std(x1)
        self.x2_mean, self.x2_std = np.mean(x2), np.std(x2)
        
        if self.debug:
            print("Data statistics:")
            print("First series (x1):")
            print(f"Mean: {self.x1_mean:.6f}, Std: {self.x1_std:.6f}")
            print("Second series (x2):")
            print(f"Mean: {self.x2_mean:.6f}, Std: {self.x2_std:.6f}")
    
    def transform_parameters(self, params):
        """Transform unconstrained parameters to model parameters with improved stability"""
        try:
            # Means relative to data means
            mu_x1_low = self.x1_mean + params[0] * self.x1_std
            mu_x1_high = self.x1_mean + params[1] * self.x1_std
            mu_x2_low = self.x2_mean + params[2] * self.x2_std
            mu_x2_high = self.x2_mean + params[3] * self.x2_std
            
            # Variances relative to data variance
            sigma_x1_low = self.x1_std * np.exp(params[4])
            sigma_x1_high = self.x1_std * np.exp(params[5])
            sigma_x2_low = self.x2_std * np.exp(params[6])
            sigma_x2_high = self.x2_std * np.exp(params[7])
            
            # Transition probabilities with strong persistence
            eps = 1e-6
            p_mu_11 = 0.5 + 0.49 * np.tanh(params[8])
            p_mu_22 = 0.5 + 0.49 * np.tanh(params[9])
            p_sigma_11 = 0.5 + 0.49 * np.tanh(params[10])
            p_sigma_22 = 0.5 + 0.49 * np.tanh(params[11])
            
            # Ensure high state is actually higher than low state
            if mu_x1_high < mu_x1_low:
                mu_x1_high, mu_x1_low = mu_x1_low, mu_x1_high
            if mu_x2_high < mu_x2_low:
                mu_x2_high, mu_x2_low = mu_x2_low, mu_x2_high
            if sigma_x1_high < sigma_x1_low:
                sigma_x1_high, sigma_x1_low = sigma_x1_low, sigma_x1_high
            if sigma_x2_high < sigma_x2_low:
                sigma_x2_high, sigma_x2_low = sigma_x2_low, sigma_x2_high
            
            # Construct transition matrices
            P_mu = np.array([[p_mu_11, 1-p_mu_11],
                           [1-p_mu_22, p_mu_22]])
            
            P_sigma = np.array([[p_sigma_11, 1-p_sigma_11],
                              [1-p_sigma_22, p_sigma_22]])
            
            return (mu_x1_low, mu_x1_high, sigma_x1_low, sigma_x1_high,
                   mu_x2_low, mu_x2_high, sigma_x2_low, sigma_x2_high,
                   P_mu, P_sigma)
            
        except Exception as e:
            if self.debug:
                print(f"Parameter transformation error: {str(e)}")
            return None
    
    def hamilton_filter(self, params):
        """Implement Hamilton filter with improved numerical stability"""
        try:
            param_tuple = self.transform_parameters(params)
            if param_tuple is None:
                return np.inf, None
                
            (mu_x1_low, mu_x1_high, sigma_x1_low, sigma_x1_high,
             mu_x2_low, mu_x2_high, sigma_x2_low, sigma_x2_high,
             P_mu, P_sigma) = param_tuple
            
            # Joint transition matrix
            P = np.kron(P_mu, P_sigma)
            log_P = np.log(P + 1e-300)
            
            n_states = 4
            log_prob_t = np.log(np.ones(n_states) / n_states)
            
            T = len(self.x1)
            filtered_probs = np.zeros((T, n_states))
            log_likelihood = 0.0
            
            # Pre-compute parameters for each state
            mus_x1 = np.array([mu_x1_low, mu_x1_low, mu_x1_high, mu_x1_high])
            sigmas_x1 = np.array([sigma_x1_low, sigma_x1_high, sigma_x1_low, sigma_x1_high])
            mus_x2 = np.array([mu_x2_low, mu_x2_low, mu_x2_high, mu_x2_high])
            sigmas_x2 = np.array([sigma_x2_low, sigma_x2_high, sigma_x2_low, sigma_x2_high])
            
            for t in range(T):
                try:
                    # Standardized innovations for numerical stability
                    z_x1 = (self.x1[t] - mus_x1) / sigmas_x1
                    z_x2 = (self.x2[t] - mus_x2) / sigmas_x2
                    
                    # Log densities with overflow protection
                    log_dens_x1 = -0.5 * np.log(2 * np.pi) - np.log(sigmas_x1) - 0.5 * z_x1**2
                    log_dens_x2 = -0.5 * np.log(2 * np.pi) - np.log(sigmas_x2) - 0.5 * z_x2**2
                    
                    log_joint = log_dens_x1 + log_dens_x2 + log_prob_t
                    log_sum = logsumexp(log_joint)
                    
                    if np.isfinite(log_sum):
                        log_likelihood += log_sum
                        filtered_probs[t] = np.exp(log_joint - log_sum)
                        log_prob_t = logsumexp(log_joint[:, None] + log_P - log_sum, axis=0)
                    else:
                        return np.inf, None
                        
                except Exception as e:
                    if self.debug:
                        print(f"Error at time {t}: {str(e)}")
                    return np.inf, None
            
            return -log_likelihood, filtered_probs
            
        except Exception as e:
            if self.debug:
                print(f"Hamilton filter error: {str(e)}")
            return np.inf, None
    
    def fit(self, initial_params=None):
        """Fit model with multiple optimization attempts"""
        if initial_params is None:
            # Initialize relative to data moments
            initial_params = np.array([
                -0.5,    # μ_x1_low (in std units)
                0.5,     # μ_x1_high
                -0.5,    # μ_x2_low
                0.5,     # μ_x2_high
                -0.5,    # log(σ_x1_low/σ_x1)
                0.5,     # log(σ_x1_high/σ_x1)
                -0.5,    # log(σ_x2_low/σ_x2)
                0.5,     # log(σ_x2_high/σ_x2)
                2.0,     # tanh transform of p_μ_11
                2.0,     # tanh transform of p_μ_22
                2.0,     # tanh transform of p_σ_11
                2.0      # tanh transform of p_σ_22
            ])
        
        # Try different optimization methods
        methods = ['L-BFGS-B', 'SLSQP', 'trust-constr']
        best_result = None
        best_value = np.inf
        
        for method in methods:
            try:
                result = minimize(
                    lambda x: self.hamilton_filter(x)[0],
                    initial_params,
                    method=method,
                    options={'maxiter': 2000, 'maxfun': 2000}
                )
                
                if result.success and result.fun < best_value:
                    best_result = result
                    best_value = result.fun
                    
            except Exception as e:
                if self.debug:
                    print(f"Optimization failed with {method}: {str(e)}")
                continue
        
        if best_result is None:
            print("Warning: All optimization attempts failed")
            return None
            
        # Compute final values
        _, filtered_probs = self.hamilton_filter(best_result.x)
        params = self.transform_parameters(best_result.x)
        
        if params is None:
            return None
            
        return {
            'parameters': {
                'mu_x1_low': params[0],
                'mu_x1_high': params[1],
                'sigma_x1_low': params[2],
                'sigma_x1_high': params[3],
                'mu_x2_low': params[4],
                'mu_x2_high': params[5],
                'sigma_x2_low': params[6],
                'sigma_x2_high': params[7],
                'P_mu': params[8],
                'P_sigma': params[9]
            },
            'filtered_probabilities': filtered_probs,
            'optimization_result': best_result
        }
    
    



def save_estimates_to_excel(results, filename='model_estimates.xlsx'):
    # Create a list to store the parameters and their values
    estimates = []
    
    # Extract parameters
    params = results['parameters']
    for param_name, value in params.items():
        # Handle transition matrices
        if isinstance(value, np.ndarray):  # For P_mu and P_sigma
            estimates.append({'Parameter': f'{param_name}[0,0]', 'Value': value[0,0]})
            estimates.append({'Parameter': f'{param_name}[0,1]', 'Value': value[0,1]})
            estimates.append({'Parameter': f'{param_name}[1,0]', 'Value': value[1,0]})
            estimates.append({'Parameter': f'{param_name}[1,1]', 'Value': value[1,1]})
        else:  # For scalar parameters
            estimates.append({'Parameter': param_name, 'Value': value})
    
    # Convert to DataFrame
    df = pd.DataFrame(estimates)
    
    # Save to Excel
    df.to_excel(filename, index=False)
                        

def calculate_conditional_moments(filtered_probs, results):
    # Extract parameters
    params = results['parameters']
    mu_low = params['mu_x1_low']
    mu_high = params['mu_x1_high']
    sigma_low = params['sigma_x1_low']
    sigma_high = params['sigma_x1_high']
    
    # Calculate probability of high mean state
    prob_high_mean = filtered_probs[:, 2] + filtered_probs[:, 3]
    prob_high_vol = filtered_probs[:, 1] + filtered_probs[:, 3]
    
    # Calculate conditional mean
    conditional_mean = prob_high_mean * mu_high + (1 - prob_high_mean) * mu_low
    
    # Calculate conditional volatility
    conditional_sigma = prob_high_vol * sigma_high + (1 - prob_high_vol) * sigma_low
    
    # Create DataFrame with results
    cond_moments = pd.DataFrame({
        'Conditional_Mean': conditional_mean,
        'Conditional_Volatility': conditional_sigma,
        'Prob_High_Mean': prob_high_mean,
        'Prob_High_Vol': prob_high_vol
    })
    
    return cond_moments





