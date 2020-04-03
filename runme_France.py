# -*- coding: utf-8 -*-
import sys
import datetime
import pickle
import bayescovid19.theta_sampling as thsa

def run(load_results = False):
    # run everything
    
    if load_results:
        # load saved results
        with open('results_France.sav', 'rb') as results_file:
            results = pickle.load(results_file)

        config = results['config']
        model = results['model']
        data = results['data']
        sampler = results['sampler']

    else: 
        # Set configuration
        run_parallelized = True

        # To run MCMC in parallel under Windows, code must be put under a "if __name__ == "__main__:" statement
        if run_parallelized and sys.platform == "Win32":
            sys.exit('Please, replace "if True:" by "if __name__ == "__main__":"')

        t0_refdate = datetime.date(2020, 1, 1)    # reference date
        cutoff_time = (datetime.date(2020, 3, 18) - t0_refdate).days
        lift_time = (datetime.date(2020, 8, 1) - t0_refdate).days

        theta_initial_guess = {'R0' : 10.7,         # Basic Reproduction Rate
                               'beta_cut' : 0.45,   # Beta cut with lockdown
                               'Tinf' : 15,         # Infection Time
                               'Tinc' : 3.5,        # Incubation Time
                               'pfatal' : 0.0005,   # Death proportion for I compartment
                               't0' : 25}           # starting day of the epidemic from t0_refdate

        config = {'country': 'France',
                  'N' : 65e6,                     # Population size
                  'Gamma' : 0,                    # Parameter of vital dynamics: births
                  'mu' : 0,                       # Parameter of vital dynamics: Death rate
                  't0_refdate' : t0_refdate,
                  'cutoff_time' : cutoff_time,
                  'lift_time' : lift_time,
                  'model_type' : 'SEIRD_with_cutoff',  # model_type
                  'fatalities_treshold' : 5,      # fatalities treshold
                  'sim_duration' : 250,           # simulation duration
                  'sim_step' : 1e-1,
                  'parallel_mcmc'  : run_parallelized,
                  'ncpu' : 4,
                  'emcee_nwalkers' : 48,
                  'mcmc_steps' : 10000,
                  'data_already_downloaded': False,
                  'sigma_obs_factor1' : 30,
                  'sigma_obs_factor2' : 1,
                  'theta0' : theta_initial_guess,
                  'debug' : False,
                  'save_results' : True}

        # Set params
        theta0 = thsa.set_initial_theta(config)

        # Operations
        data, model, sampler = thsa.run(config, theta0)

        # Save results?
        if config['save_results']:
            results = {'config':config, 'model':model, 'data': data, 'sampler':sampler}

            with open('results_France.sav', 'wb') as results_file:
                pickle.dump(results, results_file)

        return config, model, data, sampler

## Post-processing
if __name__ == '__main__':

    config, model, data, sampler = run()
    
    plot_params = {'show_mcmc_chains': True,
                   'show_posterior_distribution': True,
                   'show_predictions': True,
                   'show_posterior_mean': True,
                   'show_inf': True,
                   'show_fat': True,
                   'semilogy': True}
    
    thsa.post_processing(config, model, data, sampler, plot_params)

