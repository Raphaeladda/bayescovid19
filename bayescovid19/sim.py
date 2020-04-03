#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Documentation

"""
__author__ = "R. Adda, M.Charpentier, E. Vazquez"
__copyright__ = "CentraleSupelec, 2020"
__license__ = "MIT"
__maintainer__ = "E. Vazquez"
__email__ = "emmanuel.vazquez@centralesupelec.fr"
__status__ = "alpha"

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode


def SEIRD(t, x, theta, const_params):
    # Computes...
    # 
    # Inputs:
    #   t
    #   x
    #   theta = [R, T_inf, T_inc , pfatal]
    #   ...
    # Output:
    #   dx
    # ...
    # ...

    R0 = theta[0]    # Basic Reproduction Rate
    Tinf = theta[1]  # Infection Time
    Tinc = theta[2]  # Incubation Time
    pfatal = math.pow(10, theta[3])  # Death proportion for I compartment

    N = const_params[0]
    Gamma = const_params[1]
    mu = const_params[2]

    S = x[0]
    E = x[1]
    I = x[2]
    R = x[3]
    D = x[4]

    gamma = 1 / Tinf
    a = 1 / Tinc
    beta = gamma * R0

    dS = Gamma - mu * S - beta / N * I * S
    dE = beta / N * I * S - (mu + a) * E
    dI = a * E - gamma * (1 + pfatal) * I
    dR = gamma * I - mu * R
    dD = gamma * pfatal * I

    dx = [dS, dE, dI, dR, dD]

    return dx


def SEIRD_with_cutoff(t, x, theta, const_params):
    # Computes...
    # 
    # Inputs:
    #   t
    #   x
    #   theta = [R, beta_cut, T_inf, T_inc , pfatal]
    #   ...
    # Output:
    #   dx
    # ...
    # ...

    R0 = theta[0]    # Basic Reproduction Rate
    beta_cut = theta[1]
    Tinf = theta[2]  # Infection Time
    Tinc = theta[3]  # Incubation Time
    pfatal = pow(10, theta[4])  # Death proportion for I compartment

    N = const_params[0]
    Gamma = const_params[1]
    mu = const_params[2]
    cutoff_time = const_params[3]
    lift_time = const_params[4]

    S = x[0]
    E = x[1]
    I = x[2]
    R = x[3]
    D = x[4]

    gamma = 1 / Tinf
    a = 1 / Tinc
    if t >= cutoff_time and t < lift_time:
        beta = beta_cut * gamma * R0
    else:
        beta = gamma * R0

    dS = Gamma - mu * S - beta / N * I * S
    dE = beta / N * I * S - (mu + a) * E
    dI = a * E - gamma * (1 + pfatal) * I
    dR = gamma * I - mu * R
    dD = gamma * pfatal * I

    dx = [dS, dE, dI, dR, dD]

    return dx


class model:
    '''
    Attributes
        model_type : 
        f : RHS of the ODE
        ...
        ...
        theta : parameters
        t : time axis for simulation
        x : [S (Susceptible), E (Exposed), I (Infectious), R (Recovered), D (Death)] along time
        
    Methods
        simulate
        ...
        plot
    '''

    def __init__(self, config, tobs):
        # Initialization
        # 1. Model choice
        self.set_model_type(config)

        # 2. set regional parameters
        self.set_regional_params(config)
        
        # 3. simulation parameters
        self.set_simulation_params(config)

        # 4. build t and preallocate x
        self.prepare_simulations(config, tobs)
        
    def set_model_type(self, config):
        # Model type
        self.model_type = config['model_type']

        if self.model_type == 'SEIRD':
            self.f = SEIRD
            self.state_dim = 5  # state dim
            self.theta_dim = 5  # theta dim

        elif self.model_type == 'SEIRD_with_cutoff':
            self.f = SEIRD_with_cutoff
            self.state_dim = 5  # state dim
            self.theta_dim = 6  # theta dim
        else:
            raise('model type is not implemented')
        
        
    def set_regional_params(self, config):
        # Regional parameters
        if self.model_type == 'SEIRD':
            self.regional_params = [config['N'], config['Gamma'], config['mu']]
        elif self.model_type == 'SEIRD_with_cutoff':
            self.regional_params = [config['N'], config['Gamma'], config['mu'],
                                    config['cutoff_time'],
                                    config['lift_time']]

            
    def set_simulation_params(self, config):
        # Simulation parameters
        self.time_step = config['sim_step']
        self.sim_duration = config['sim_duration']


    def prepare_simulations(self, config, tobs):
        # build t and preallocate x        
        self.t = np.arange(0, self.sim_duration, self.time_step)
        self.x = np.zeros([self.t.shape[0], self.state_dim])  # preallocation

        # start date, assume a a datetime object for t0_ref and convert to numpy datetime object
        self.t0_refdate = np.datetime64(config['t0_refdate'])

        # assume a pandas dataframe of timestamp objects and convert to numpy datetime objects
        tobs_rel = [tobs[i] - self.t0_refdate for i in range(tobs.shape[0])]
        # pdb.set_trace()

        # build a tobs_idx array in order to retrieve observations with self.x[self.tobs_idx]
        self.tobs_rel = [tobs_rel[i].days for i in range(len(tobs_rel))]
        k = 0
        self.tobs_idx = []
        for i in range(len(self.tobs_rel)):
            t = self.tobs_rel[i]
            while k < self.t.shape[0]:
                if self.t[k] >= t:
                    self.tobs_idx.append(k)
                    break
                k += 1
        # pdb.set_trace()

    def y_from_tobs(self):
        # Helper function to retrieve fatalities at tobs
        fatalities_idx = 4
        return self.x[self.tobs_idx, fatalities_idx]

    def simulate(self, theta):
        # ODE integration happens here

        # Set theta
        self.theta = theta

        # State initialization
        t0 = theta[-1]
        x0 = [self.regional_params[0] - 1, 0, 1, 0, 0]

        # ODE initialization
        # r = ode(f).set_integrator('vode', method='adams', with_jacobian=False)
        r = ode(self.f).set_integrator('lsoda', with_jacobian=False)
        r.set_initial_value(x0, t0).set_f_params(self.theta, self.regional_params)

        idx_start = 0
        while self.t[idx_start] < t0:
            idx_start += 1

        # Run simulation
        for i in range(idx_start + 1, self.t.shape[0]):
            # NB1: r.successful() is not tested
            # NB2: r.t is self-incrementing
            t = self.t[i]
            r.integrate(t)
            self.x[i, :] = r.y

    def plot(self, semilog=True):
        # Simple plot
        if semilog:
            plt.semilogy(self.t, self.x[:, 0], label="Susceptible")
            plt.semilogy(self.t, self.x[:, 1], label="Exposed")
            plt.semilogy(self.t, self.x[:, 2], label="Infectious")
            plt.semilogy(self.t, self.x[:, 3], label="Recovered")
            plt.semilogy(self.t, self.x[:, 4], label="Death")
            plt.ylim((0.1, np.max(self.x[:, 0])))
        else:
            plt.plot(self.t, self.x[:, 1], label="Exposed")
            plt.plot(self.t, self.x[:, 2], label="Infectious")
            plt.plot(self.t, self.x[:, 4], label="Death")

        plt.title("Propagation model SEIR(D)")
        plt.grid(True)

        plt.legend()
        plt.show()

    def plot_with_obs(self, yobs, ax=None, transparency=1):
        # Simple plot
        if ax is None:
            fig, ax = plt.subplots()
            plt.xlabel('Time (days)')
            plt.ylabel(r'$\log_{10}(y(t))$')

        idx = np.where(yobs > 0)
        tobs = np.array(self.tobs_rel)
        tobs = tobs[idx[0]]
        yobs = yobs[idx[0]]

        # pdb.set_trace()
        plt.semilogy(self.t, self.x[:, 2], label='Prediction', color='C4', alpha=transparency / 2)
        plt.semilogy(self.t, self.x[:, 4], label='Prediction', color='C7', alpha=transparency)
        plt.semilogy(tobs, yobs, label='Observed', linestyle='dashed', marker='o', color='C1')
        return ax
