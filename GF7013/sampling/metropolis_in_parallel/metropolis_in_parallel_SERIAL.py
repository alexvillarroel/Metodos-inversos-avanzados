"""
Metropolis Algorithm

Generates samples from a pdf using the Metropolis Algorithm

Francisco Ortega Culaciati
ortega.francisco@uchile.cl
GF7013 - Metodos Inversos Avanzados
Departamento de Geofisica - FCFM - Universidad de Chile 

"""
COMPLETAR = None
import numpy as NP
from ...model_parameters import ensemble
from ..metropolis import metropolis

def metropolis_in_parallel_SERIAL(m0, likelihood_fun, pdf_prior, proposal, num_MCMC_steps,  
               use_log_likelihood = True):
    """
    Performs the Metropolis in Parallel Algorithm. THIS IS THE SERIAL VERSION OF THE 
    ALGORITHM, THUS IT RUNS ONE MCMC CHAIN AT EACH TIME.

    - m0 : initial ensemble of models for the MCMC chains.
    - likelihood_fun: an object that provides the functions likelihood_fun.likelihood(m)
                      and likelihood_fun.log_likelihood(m) that return the value or 
                      natural logarithm of the value of the likelihood function used
                      for the inverse problem.
    - pdf_prior: an object that provides the functions fprior.likelihood(m) and 
                      fprior.log_likelihood(m) that return the value or 
                      natural logarithm of the value of the prior probability
                      function on model parameters used for the inverse problem.
    - proposal: an object that provides the function proposal.propose(m) that returns
               a model m_test, proposed as the next step of the MCMC chain.
    - num_MCMC_steps: Number of MCMC steps of each Metropolis algorithm produced in 
                      parallel.
    - use_log_likelihood: if True, uses the logarithm of the likelihood of fprior and
                         of likelihood_fun to evaluate the acceptance probabilities
                         in Metropolis algorithm. Thus, needs fprior.log_likelihood(m)
                         and likelihood_fun.log_likelihood(m). If False, uses the actual
                         likelihood values, computed from fprior.likelihood(m) and 
                         likelihood_fun.likelihood(m) to evaluate acceptance probability.
    NOTE: the exponent beta of the likelihood function for TMCMC algorithm must be defined
          in the initial ensemble of models (use beta=1 if not performing TMCMC).
                          
    """
    # Number of models in the initial ensemble.
    Npar = m0.Npar 
    Nmodels = m0.Nmodels

    # N burn-in iterations.
    num_burnin = num_MCMC_steps - 1
    num_samples = 1

    # Emsemble of models to be returned after the MCMC steps.
    m = ensemble(Npar, Nmodels, use_log_likelihood = use_log_likelihood) 
    acceptance_ratios = NP.zeros(Nmodels)  # Acceptance ratios for each MCMC chain.


    # Loop over the initial ensemble of models.
    for i in range(Nmodels):
        # Using the Metropolis algorithm to perform MCMC steps in parallel.
        # Each model in the initial ensemble is treated as a separate MCMC chain.

        m0_i = m0.m_set[i, :]  # Select the i-th model from the initial ensemble.

        # Run the Metropolis algorithm for the i-th model. 
        results = metropolis(m0_i, likelihood_fun, pdf_prior, proposal, num_samples, 
                             num_burnin, use_log_likelihood = use_log_likelihood, 
                             save_samples = None, beta = 1, LogOfZero = None, 
                             rng = None, seed = None)
        
        # Store the results in the ensemble.
        m.m_set[i, :] = results["m"]
        m.fprior[i] = results["fprior"]
        m.like[i] = results["like"]
        m.f[i] = results["fpost"]
        acceptance_ratios[i] = results["acceptance_ratio"]

    return m, acceptance_ratios

