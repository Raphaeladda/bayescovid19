# -*- coding: utf-8 -*-
__author__ = "M. Charpentier, E. Vazquez"
__copyright__ = "CentraleSupelec, 2020"
__license__ = "MIT"
__maintainer__ = "E. Vazquez"
__email__ = "emmanuel.vazquez@centralesupelec.fr"
__status__ = "alpha"

import math
import os
from multiprocessing import Pool

import emcee
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from tqdm import tqdm

from . import sim
from .import_data import get_fatalities
from .likelihood import log_posterior


def run(config, theta0):
    # -------- Main operations -------- 

    print('''
    This program performs predictions of the Covid-19 epidemy using a
    S(E)IRD model and a Bayesian approach. Data are downloaded from
    public repositories (ECDC and Santé Publique France)
    ''')
    print('Getting data...', end='')
    data = get_data(config)
    print(' done')
    
    print('Building model...', end='')
    model = init_model(config, data, theta0)
    print(' done')
    
    print('Infering standard deviation of observations from the model...')
    data, theta1 = infer_std(config, data, model)
    print('... done')

    print('Running MCMC...')
    sampler = run_mcmc_emcee(theta1, data, model, config)
    print('... done')
    
    return data, model, sampler


def get_data(config):
    # -------- Get data and put them in a Pandas dataframe --------

    df_fatalities = get_fatalities(config['country'], reuse=config['data_already_downloaded'])

    if 'regions' in config and config['regions']:
        total = df_fatalities[ config['regions'] ].sum(axis = 1)
        data = df_fatalities[['date', 'total']].copy()
        data['total'] = total
    else:
        data = df_fatalities[['date', 'total']].copy()
    data.columns = ['tobs', 'yobs']

    return data


def init_model(config, data, theta0):
    # Set model
    tobs_df = data['tobs'].copy()
    model = sim.model(config, tobs_df)

    if config['debug']:
        model.simulate(theta0)
        model.plot_with_obs(data['yobs'].values)
        plt.title('Initial guess, theta0')
        plt.show()

    return model


def run_mcmc_emcee(theta0, data, model, config):
    ## MCMC
    
    threshold = config['fatalities_treshold']
    logpdf = lambda theta: log_posterior(theta, data, model, threshold)

    nwalkers = config['emcee_nwalkers']
    dim = theta0.shape[0]
    pos = theta0 + 0.1 * np.random.randn(nwalkers, dim)
    nwalkers, dim = pos.shape

    parallel = config['parallel_mcmc']
    if parallel:
        os.environ["OMP_NUM_THREADS"] = str(config['ncpu'])

        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers,
                                            dim,
                                            log_posterior,
                                            args=(data, model, threshold),
                                            pool=pool)
            sampler.run_mcmc(pos, config['mcmc_steps'], progress=True)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, dim, logpdf)
        sampler.run_mcmc(pos, config['mcmc_steps'], progress=True)

    return sampler


def set_initial_theta(config):
    # Set theta and regional_params
    
    if config['model_type'] == 'SEIRD':
        R0 = config['theta0']['R0']      # Basic Reproduction Rate
        Tinf = config['theta0']['Tinf']  # Infection Time
        Tinc = config['theta0']['Tinc']  # Incubation Time
        pfatal = config['theta0']['pfatal']  # Death proportion for I compartment
        t0 = config['theta0']['t0']      # starting day of the epidemic from t0_refdate
        
        theta0 = np.array([R0, Tinf, Tinc, math.log10(pfatal), t0])

    elif config['model_type'] == 'SEIRD_with_cutoff':
        R0 = config['theta0']['R0']      # Basic Reproduction Rate
        beta_cut = config['theta0']['beta_cut']  # Beta cut with lockdown
        Tinf = config['theta0']['Tinf']  # Infection Time
        Tinc = config['theta0']['Tinc']  # Incubation Time
        pfatal = config['theta0']['pfatal']  # Death proportion for I compartment
        t0 = config['theta0']['t0']      # starting day of the epidemic from t0_refdate

        theta0 = np.array([R0, beta_cut, Tinf, Tinc, math.log10(pfatal), t0])

    elif config['model_type'] == 'SIRD_with_cutoff':
        R0 = config['theta0']['R0']      # Basic Reproduction Rate
        beta_cut = config['theta0']['beta_cut']  # Beta cut with lockdown
        Tinf = config['theta0']['Tinf']  # Infection Time
        pfatal = config['theta0']['pfatal']  # Death proportion for I compartment
        t0 = config['theta0']['t0']      # starting day of the epidemic from t0_refdate

        theta0 = np.array([R0, beta_cut, Tinf, math.log10(pfatal), t0])

    return theta0

def infer_std(config, data, model):
    # Ideally, the likelihood of the observations is Poisson, so that
    # the variances of the deviations of the observations yobs from
    # the model could be approximated by the yobs. Unfortunately, the
    # data may be corrupted. We replace the Poisson likelihood with a
    # Gaussian likelihood and rely on a 2-step procedure to estimate
    # the variance.
    
    # Step 1
    
    sigma = np.zeros([data.shape[0]])
    yobs = data['yobs'].values.reshape(-1) # pd.dataframe > np.array
    sigma = np.sqrt(yobs)
    data['sigma'] = sigma
    
    theta1_ = map_estimation(config, data, model)

    # Step 2

    model.simulate(theta1_)
    ypred = model.y_from_tobs()
    
    idx = data['yobs'].ge(config['fatalities_treshold'])
    yobs = data['yobs'][idx].values.reshape(-1) # pd.dataframe > np.array
    ypred = ypred[idx]
    ypredlog10 = np.log10(ypred)
    s2log10 = np.log10((yobs - ypred) ** 2)
    
    f = lambda x: np.mean( (x[0] * ypredlog10 + x[1] - s2log10)**2 )
    g = lambda x: np.min(x[0] * ypredlog10 + x[1] - s2log10)
    con = {'type': 'ineq', 'fun': g}
    x0 = [0, -1]
    soln = minimize(f, x0, method='SLSQP', constraints=con)
    x_opt = soln.x
    s2log10up = x_opt[0] * ypredlog10 + x_opt[1]
    sigma[idx] = np.sqrt(np.power(10, s2log10up))
    data['sigma'] = sigma

    if config['debug'] == True:
        plt.loglog(ypred, (yobs - ypred)**2, 'o', ypred, np.power(10, s2log10up))
        plt.title('Upper bound on the squares of the deviations of the observations from the model')
        plt.show()

    theta1 = map_estimation(config, data, model)

    return data, theta1
    
def map_estimation(config, data, model):
    # Maximum a posteriori estimation
    
    nll = lambda *args: -log_posterior(*args)

    theta0 = set_initial_theta(config)

    soln = minimize(nll, theta0, args=(data, model, 5), method='Nelder-Mead', options={'disp': config['debug']})

    theta_MAP = soln.x

    if config['debug']:
        print(theta_MAP)
        model.simulate(theta_MAP)
        #model.plot_with_obs(data['yobs'].values)

        fig, ax = plt.subplots()
        plt.xlabel('Time (days)')
        plt.ylabel(r'$\log_{10}(y(t))$')

        idx = data['yobs'].gt(0)      
        tobs = np.array(model.tobs_rel)[idx]
        yobs = data['yobs'][idx]
        sigma = data['sigma'][idx]
        
        plt.semilogy(model.t, model.x[:, model.state_ref['I']], label='Prediction', color='C4', alpha= 1)
        plt.semilogy(model.t, model.x[:, model.state_ref['D']], label='Prediction', color='C7', alpha= 1)
        plt.semilogy(tobs, yobs, label='Observed', linestyle='dashed', marker='o', color='C1')
        plt.semilogy(tobs, np.maximum(1e-1, yobs - 2*sigma), '--', color='k', alpha=0.3)
        plt.semilogy(tobs, yobs + 2*sigma , '--', color='k', alpha=0.3)

        v = ["%.3f" % theta_MAP[i] for i in range(theta_MAP.shape[0])]
        title = 'Best fit, theta_MAP=' + str(v)
        plt.title(title)
        plt.show()

    return theta_MAP
