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
from multiprocessing import Pool, cpu_count


def metropolis_in_parallel_POOL(m0, likelihood_fun, pdf_prior, proposal, num_MCMC_steps,  
               use_log_likelihood = True):
    """
    Performs the Metropolis in Parallel Algorithm. THIS IS A PARALLELIZED VERSION OF THE 
    ALGORITHM, THUS IT RUNS AS MANY MCMC CHAINS AT THE SAME TIME AS THE NUMBER OF 
    COMPUTING PROCESSORS THAT ARE AVAILABLE. THIS CODE USES multiprocessing.Pool
    https://docs.python.org/3/library/multiprocessing.html

    REMEMBER TO SET THIS ENVIRONMENT VARIABLES IN THE MAIN SCRIPT OF THE CODE BEFORE
    IMPORTING ANY OTHER PACKAGE
    # set the number of threads for numpy
      import os
      os.environ["OMP_NUM_THREADS"] = "1"
      os.environ["MKL_NUM_THREADS"] = "1"
      os.environ["MKL_DOMAIN_NUM_THREADS"] = "1"
      os.environ["OPENBLAS_NUM_THREADS"] = "1"
      os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
      os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
                         likelihood_fun to evaluate the acceptance probabilities
                         in Metropolis algorithm. Thus, needs fprior.log_likelihood(m)
                         and likelihood_fun.log_likelihood(m). If False, uses the actual
                         likelihood values, computed from fprior.likelihood(m) and 
                         likelihood_fun.likelihood(m) to evaluate acceptance probability.
                         
    NOTE:-the exponent beta of the likelihood function for TMCMC algorithm must be defined
          in the initial ensemble of models. Use beta=1 if not performing TMCMC or beta=0 
          in the initial ensemble if performing TMCMC.
         -BE careful with random number generation. Each parallel process must have a 
         different seed.                
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
    save_samples = False  # Do not save samples in the ensemble, only the final model
    beta = m0.beta  # Use the beta value defined in the initial ensemble.
    LogOfZero = None
    rng = None
    seed = None

    # List of arguments for each MCMC chain to be run in parallel.
    args_list = [ (m0.m_set[i, :], likelihood_fun, pdf_prior, proposal, num_samples, num_burnin, use_log_likelihood, 
                    save_samples, beta, LogOfZero, rng, seed) for i in range(Nmodels)] # Lista de argumentos por cada modelo

    # Use multiprocessing.Pool to run the Metropolis algorithm in parallel.
    with Pool(processes=cpu_count()) as pool:
        
      results_pool = pool.starmap(metropolis, args_list) # Retorna una lista
    
    # Store the results in the ensemble.
    for i, result in enumerate(results_pool):
      m.m_set[i, :] = result["m"]
      m.fprior[i] = result["fprior"]
      m.like[i] = result["like"]
      m.f[i] = result["fpost"]
      acceptance_ratios[i] = result["acceptance_ratio"]
    
    return m, acceptance_ratios

