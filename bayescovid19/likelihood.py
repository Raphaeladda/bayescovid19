# -*- coding: utf-8 -*-

""" Documentation

"""
__author__ = "M.Charpentier, E. Vazquez"
__copyright__ = "CentraleSupelec, 2020"
__license__ = "MIT"
__maintainer__ = "E. Vazquez"
__email__ = "emmanuel.vazquez@centralesupelec.fr"
__status__ = "alpha"

import numpy as np


def log_likelihood(theta, data, model, threshold=5):
    '''
    Inputs
        data = [T,Y]
        theta = [R0, Tinf, Tinc, pfatal]
        param = [N, Gamma, mu, sigma]
    Outputs
        lP = log_likelihood logp(Y|theta)
        Y = simulated vector
    '''

    # Observations

    y_obs     = data['yobs'].values.reshape(-1)
    sigma_obs = data['sigma'].values.reshape(-1)
    
    # if number of fatalities < 10 the model is innacurate
    k0 = 0
    while y_obs[k0] < threshold:
        k0 += 1

    # Simulation
    model.simulate(theta)
    y_pred = model.y_from_tobs()

    # We use a Poisson likelihood (approximated using a Gaussian distribution)
    loglik = - 1 / 2 * np.sum(1 / sigma_obs[k0:]**2 * (y_obs[k0:] - y_pred[k0:]) ** 2)
    
    return loglik


def log_prior_SEIRD(theta):
    # intervals correct?

    R0 = theta[0]  # Basic Reproduction Rate
    Tinf = theta[1]  # Infection Time
    Tinc = theta[2]  # Incubation Time
    log10pfatal = theta[3]  # Death proportion for I compartment
    t0 = theta[4]  # Start time

    if R0 < 0 or R0 > 200:
        return -np.inf
    elif Tinf < 1 or Tinf > 200:
        return -np.inf
    elif Tinc < 1 or Tinc > 200:
        return -np.inf
    elif log10pfatal < -6 or log10pfatal > -1:
        return -np.inf
    elif t0 < 1 or t0 > 100:
        return -np.inf
    else:
        return 0


def log_prior_SEIRD_with_cutoff(theta):
    # intervals correct?

    R0 = theta[0]  # Basic Reproduction Rate
    beta_cut = theta[1]  # Reproduction Rate during lockdown
    Tinf = theta[2]  # Infection Time
    Tinc = theta[3]  # Incubation Time
    log10pfatal = theta[4]  # Death proportion for I compartment
    t0 = theta[5]  # Start time

    if R0 < 0 or R0 > 120:
        return -np.inf
    elif beta_cut < 0 or beta_cut > 1:
        return -np.inf
    elif Tinf < 1 or Tinf > 90:
        return -np.inf
    elif Tinc < 1 or Tinc > 16:
        return -np.inf
    elif log10pfatal < -5 or log10pfatal > -2:
        return -np.inf
    elif t0 < 1 or t0 > 100:
        return -np.inf
    else:
        return 0


def log_prior_SIRD_with_cutoff(theta):
    # intervals correct?

    R0 = theta[0]  # Basic Reproduction Rate
    beta_cut = theta[1]  # Reproduction Rate during lockdown
    Tinf = theta[2]  # Infection Time
    log10pfatal = theta[3]  # Death proportion for I compartment
    t0 = theta[4]  # Start time

    if R0 < 0 or R0 > 120:
        return -np.inf
    elif beta_cut < 0 or beta_cut > 1:
        return -np.inf
    elif Tinf < 1 or Tinf > 90:
        return -np.inf
    elif log10pfatal < -5 or log10pfatal > -2:
        return -np.inf
    elif t0 < 1 or t0 > 100:
        return -np.inf
    else:
        return 0

    
def log_prior(theta, model):
    if model.model_type == 'SEIRD':
        return log_prior_SEIRD(theta)
    elif model.model_type == 'SEIRD_with_cutoff':
        return log_prior_SEIRD_with_cutoff(theta)
    elif model.model_type == 'SIRD_with_cutoff':
        return log_prior_SIRD_with_cutoff(theta)


def log_posterior(theta, data, model, threshold=5):
    lp = log_prior(theta, model)
    if not np.isfinite(lp):
        return -np.inf
    else:
        ll = log_likelihood(theta, data, model, threshold)
        return ll + lp
