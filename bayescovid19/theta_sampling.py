# -*- coding: utf-8 -*-
__author__ = "M. Charpentier, E. Vazquez"
__copyright__ = "CentraleSupelec, 2020"
__license__ = "MIT"
__maintainer__ = "E. Vazquez"
__email__ = "emmanuel.vazquez@centralesupelec.fr"
__status__ = "alpha"

import math
import os
import datetime
from multiprocessing import Pool

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from tqdm import tqdm

from . import sim
from .import_data import get_fatalities
from .likelihood import log_posterior


def run(config, theta0):
    # Operations

    print('Get data...')
    data = get_data(config)

    print('Build model...')
    model = init_model(config, data, theta0)

    print('Infer standard deviation of observations from the model')
    data, theta1 = infer_std(config, data, model)

    print('Run MCMC...')
    sampler = run_mcmc_emcee(theta1, data, model, config)

    return data, model, sampler


def get_data(config):
    # Get data and put them in a Pandas dataframe

    df_fatalities = get_fatalities(config['country'], reuse=config['data_already_downloaded'])

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

    return theta0

def infer_std(config, data, model):
    # Ideally, the likelihood of the observations is Poisson, so that
    # the variances of the deviations of the observations yobs from
    # the model could be approximated by the yobs. Unfortunately, the
    # data may be corrupted. We replace the Poisson likelihood with a
    # Gaussian likelihood and rely on a 2-step procedure to estimate
    # the variance.
    
    print('Finding a best fit by maximum a posteriori estimation...')

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

    soln = minimize(nll, theta0, args=(data, model, 5), method='Nelder-Mead', options={'disp': True})

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
        
        plt.semilogy(model.t, model.x[:, 2], label='Prediction', color='C4', alpha= 1)
        plt.semilogy(model.t, model.x[:, 4], label='Prediction', color='C7', alpha= 1)
        plt.semilogy(tobs, yobs, label='Observed', linestyle='dashed', marker='o', color='C1')
        plt.semilogy(tobs, np.maximum(1e-1, yobs - 2*sigma), '--', color='k', alpha=0.3)
        plt.semilogy(tobs, yobs + 2*sigma , '--', color='k', alpha=0.3)

        v = ["%.3f" % theta_MAP[i] for i in range(theta_MAP.shape[0])]
        title = 'Best fit, theta_MAP=' + str(v)
        plt.title(title)
        plt.show()

    return theta_MAP


def post_processing(config, model, data, sampler, plot_params):
    # View results

    samples = sampler.get_chain()
    n_discard = int(config['mcmc_steps'] * 3 / 4)
    theta_post = sampler.get_chain(discard=n_discard, thin=20, flat=True)
    
    if plot_params['show_mcmc_chains']:
        plot_samples(config, model, data, samples)    

    if plot_params['show_posterior_distribution']:
        show_posterior_distribution(config, theta_post)

    if plot_params['show_predictions']:
        show_predictions(config, model, data, theta_post, plot_params)


def plot_samples(config, model, data, samples):
    # Plot the samples of the chains
    
    labels = list(config['theta0'].keys())

    fig, axes = plt.subplots(model.theta_dim, figsize=(10, 7), sharex='all')

    for i in range(model.theta_dim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    plt.show()


def show_posterior_distribution(config, theta_post):
    # Show posterior distribution of the parameters of the model
    
    labels = list(config['theta0'].keys())

    fig = corner.corner(theta_post, labels=labels)
    plt.show()


def show_predictions(config, model, data, theta_post, plot_params):
    # Show the predictions of the model
    
    yobs = data['yobs'].values
    idx = np.where(yobs > 0)
    tobs = np.array(model.tobs_rel)
    tobs = tobs[idx[0]]
    yobs = yobs[idx[0]]

    if plot_params['show_posterior_mean'] == True:
        # compute posterior mean
        n0 = 500
        inds = np.random.randint(theta_post.shape[0], size=n0)
        nt = model.x.shape[0]
        all_inf0 = np.zeros([nt, n0])
        all_fat0 = np.zeros([nt, n0])
        all_inf1 = np.zeros([nt, n0])
        all_fat1 = np.zeros([nt, n0])
        all_inf2 = np.zeros([nt, n0])
        all_fat2 = np.zeros([nt, n0])

        print('Computing posterior mean')
        for i in tqdm(range(n0)):  # , bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            # what if there was no lockdown
            theta_post_cf = theta_post[inds[i], :].copy()
            theta_post_cf[1] = 1
            model.simulate(theta_post_cf)
            all_inf1[:, i] = model.x[:, 2]
            all_fat1[:, i] = model.x[:, 4]

            # what if lockdown is removed now
            lift_time_org = config['lift_time']
            config['lift_time'] =  (datetime.date(2020, 4, 3) - datetime.date(2020, 1, 1)).days
            model.set_regional_params(config)
            model.simulate(theta_post[inds[i], :])
            all_inf2[:, i] = model.x[:, 2]
            all_fat2[:, i] = model.x[:, 4]

            # actual situation
            config['lift_time'] = lift_time_org
            model.set_regional_params(config)
            model.simulate(theta_post[inds[i], :])
            all_inf0[:, i] = model.x[:, 2]
            all_fat0[:, i] = model.x[:, 4]

        n_inf0_posterior_mean = np.mean(all_inf0, axis=1)
        n_fat0_posterior_mean = np.mean(all_fat0, axis=1)

        n_inf1_posterior_mean = np.mean(all_inf1, axis=1)
        n_fat1_posterior_mean = np.mean(all_fat1, axis=1)

        n_inf2_posterior_mean = np.mean(all_inf2, axis=1)
        n_fat2_posterior_mean = np.mean(all_fat2, axis=1)

    # display
    fig, ax = plt.subplots()

    n1 = 100
    inds = np.random.randint(theta_post.shape[0], size=n1)
    for i in inds:
        # # what if there was no lockdown
        # theta_post_cf = theta_post[i, :].copy()
        # theta_post_cf[1] = 1
        # model.simulate(theta_post_cf)
        # plt.semilogy(model.t, model.x[:, 4], color='C4', alpha=0.3)

        # # what if lockdown is removed now
        # lift_time_org = config['lift_time']
        # config['lift_time'] =  (datetime.date(2020, 4, 3) - datetime.date(2020, 1, 1)).days
        # model.set_regional_params(config)
        # model.simulate(theta_post[i, :])
        # plt.semilogy(model.t, model.x[:, 4], color='C1', alpha=0.4)
        
        # # actual situation
        # config['lift_time'] = lift_time_org
        # model.set_regional_params(config)
        model.simulate(theta_post[i, :])
        if plot_params['show_inf'] == True:
            plt.plot(model.t, model.x[:, 2], color='C2', alpha=0.2)
        if plot_params['show_fat'] == True:
            plt.plot(model.t, model.x[:, 4], color='C0', alpha=0.5)

    if plot_params['show_posterior_mean'] and plot_params['show_inf'] == True:
        plt.plot(model.t, n_inf1_posterior_mean, label='Prediction', color='C6', alpha=1)
        plt.plot(model.t, n_inf2_posterior_mean, label='Prediction', color='C3', alpha=1)
        plt.plot(model.t, n_inf0_posterior_mean, label='Prediction', color='C1', alpha=1)
        
    if plot_params['show_posterior_mean'] and plot_params['show_fat'] == True:
        plt.plot(model.t, n_fat1_posterior_mean, label='Prediction', color='C6', alpha=1)
        plt.plot(model.t, n_fat2_posterior_mean, label='Prediction', color='C3', alpha=1)
        plt.plot(model.t, n_fat0_posterior_mean, label='Prediction', color='C1', alpha=1)

    plt.plot(tobs, yobs, label='Observed', linestyle='dashed', marker='o', color='C3')

    if plot_params['semilogy']:
        plt.xlabel('Time (days)')
        plt.ylabel(r'$\log_{10}(y(t))$')
        plt.yscale('log')
    else:
        plt.xlabel('Time (days)')
        plt.ylabel(r'$y(t)$')
        
    ax.grid()
    plt.show()
