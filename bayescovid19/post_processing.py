# -*- coding: utf-8 -*-
__author__ = "E. Vazquez"
__copyright__ = "CentraleSupelec, 2020"
__license__ = "MIT"
__maintainer__ = "E. Vazquez"
__email__ = "emmanuel.vazquez@centralesupelec.fr"
__status__ = "alpha"

import datetime
import numpy as np
import matplotlib.pyplot as plt
import corner
from tqdm import tqdm


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


def posterior_trajectories (config, model, data, theta_post, n0):
    # Draw posterior trajectories
    # n0: number of trajectories

    nt = model.x.shape[0]
    dim = model.state_dim

    traj_base = np.zeros([n0, nt, dim])
    traj_nolockdown = np.zeros([n0, nt, dim])
    traj_rmlockdown = np.zeros([n0, nt, dim])
    
    inds = np.random.randint(theta_post.shape[0], size=n0)
    print('Computing posterior mean...')

    beta_cut_idx = 1
    
    for i in tqdm(range(n0)):  # , bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        # what if no lockdown had been enforced
        theta_post_cf = theta_post[inds[i], :].copy()
        theta_post_cf[beta_cut_idx] = 1.0
        model.simulate(theta_post_cf)
        traj_nolockdown[i, :, :] = model.x
        
        # what if lockdown is removed April 15th
        lift_time_org = config['lift_time']
        config['lift_time'] =  (datetime.date(2020, 4, 15) - datetime.date(2020, 1, 1)).days
        model.set_regional_params(config)
        model.simulate(theta_post[inds[i], :])
        traj_rmlockdown[i, :, :] = model.x

        # actual situation
        config['lift_time'] = lift_time_org
        model.set_regional_params(config)
        model.simulate(theta_post[inds[i], :])
        traj_base[i, :, :] = model.x

    traj_base_stats = {}
    traj_nolockdown_stats = {}
    traj_rmlockdown_stats = {}
    # means
    traj_base_stats['mean'] = np.mean(traj_base, axis=0)
    traj_nolockdown_stats['mean'] = np.mean(traj_nolockdown, axis=0)
    traj_rmlockdown_stats['mean'] = np.mean(traj_rmlockdown, axis=0)

    # q 2.5% 
    traj_base_stats['low'] = np.quantile(traj_base, 0.025, axis=0)
    traj_nolockdown_stats['low'] = np.quantile(traj_nolockdown, 0.025, axis=0)
    traj_rmlockdown_stats['low'] = np.quantile(traj_rmlockdown, 0.025, axis=0)
    
    # q 2.5% 
    traj_base_stats['up'] = np.quantile(traj_base, 0.975, axis=0)
    traj_nolockdown_stats['up'] = np.quantile(traj_nolockdown, 0.975, axis=0)
    traj_rmlockdown_stats['up'] = np.quantile(traj_rmlockdown, 0.975, axis=0)

    return traj_base_stats, traj_nolockdown_stats, traj_rmlockdown_stats


def show_predictions(config, model, data, theta_post, plot_params):
    # Show the predictions of the model

    # Get data
    yobs = data['yobs'].values
    idx = np.where(yobs > 0)
    tobs = np.array(model.tobs_rel)
    tobs = tobs[idx[0]] * 24
    tobs = np.datetime64(config['t0_refdate']) + tobs.astype('timedelta64[h]')
    yobs = yobs[idx[0]]

    dt = model.t * 24
    t = np.datetime64(config['t0_refdate']) + dt.astype('timedelta64[h]')
    
    # Compute posterior means?
    if plot_params['show_posterior_mean'] == True:
        traj_base_stats, traj_nolockdown_stats, traj_rmlockdown_stats = posterior_trajectories(
            config, model, data, theta_post, 100)
    # display

    # Setting plot parameters
    colors = {'Canada':'#045275',
              'China':'#089099',
              'France':'#7CCBA2',
              'Germany':'#FCDE9C',
              'US':'#DC3977',
              'United Kingdom':'#7C1D6F'}

    # plot.text(x = covid.index[1], y = int(covid.max().max())+45000, s
    #           = "COVID-19 Cases by Country", fontsize = 23, weight = 'bold',
    #           alpha = .75)
    # plot.text(x = covid.index[1], y = int(covid.max().max())+15000, s
    #           = "For the USA, China, Germany, France, United Kingdom, and
    #           Canada\nIncludes Current Cases, Recoveries, and Deaths",
    #           fontsize = 16, alpha = .75)
    # plot.text(x = percapita.index[1], y = -100000,s = 'datagy.io
    # Source:
    # https://github.com/datasets/covid-19/blob/master/data/countries-aggregated.csv',
    #           fontsize = 10)
    import matplotlib.dates as mdates
    
    fig, ax = plt.subplots() #, color=list(colors.values()), linewidth=5, legend=False)
    locator = mdates.AutoDateLocator() #(minticks=3, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    
    pop_sz = model.regional_params[0]

    ## Cumulated cases
    if plot_params['show_posterior_mean']:
        plt.plot(t,
                 pop_sz - traj_nolockdown_stats['mean'][:, model.state_ref['S']],
                 linestyle='dashed', color='C2', alpha=1)

        plt.plot(t, pop_sz - traj_base_stats['mean'][:, model.state_ref['S']], label='Cumulated', color='C8', alpha=1)
        plt.fill_between(t,
                         pop_sz - traj_base_stats['low'][:, model.state_ref['S']],
                         pop_sz - traj_base_stats['up'][:, model.state_ref['S']],
                         color='C8', alpha=0.2)

    # Active cases
    if plot_params['show_posterior_mean'] and plot_params['show_inf'] == True:
        plt.plot(t,
                 traj_nolockdown_stats['mean'][:, model.state_ref['I']],
                 linestyle='dashed', color='C2', alpha=1)
        plt.plot(t, traj_base_stats['mean'][:, model.state_ref['I']], label='Active', color='C0', alpha=1)
        plt.fill_between(t,
                         traj_base_stats['low'][:, model.state_ref['I']],
                         traj_base_stats['up'][:, model.state_ref['I']],
                         color='C0', alpha=0.2)
    # Deaths
    if plot_params['show_posterior_mean'] and plot_params['show_fat'] == True:
        plt.plot(t,
                 traj_nolockdown_stats['mean'][:, model.state_ref['D']],
                 linestyle='dashed', color='C2', alpha=1)
        plt.plot(t,
                 traj_base_stats['mean'][:, model.state_ref['D']],
                 label='Deaths', color='C6', alpha=1)
        plt.fill_between(t,
                         traj_base_stats['low'][:, model.state_ref['D']],
                         traj_base_stats['up'][:, model.state_ref['D']],
                         color='C6', alpha=0.2)
        
    # Observations
    plt.plot(tobs, yobs, label='Observations', linestyle='dashed', marker='o', color='C3')

    # Population size
    left, right = plt.xlim()
    plt.plot([left, right], [pop_sz, pop_sz], color= 'C2')
    
    if plot_params['semilogy']:
        plt.xlabel('Date')
        plt.ylabel(r'# Cases (log-scale)')
        plt.yscale('log')
    else:
        plt.xlabel('Time (days)')
        plt.ylabel(r'# Cases')

    plt.legend()
    ax.grid(color='#d4d4d4')
    plt.show()
