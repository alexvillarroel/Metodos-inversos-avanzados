#! /usr/bin/env python
import sys, os
# get folder of this module
this_module_folder = os.path.dirname(os.path.abspath(__file__))
# add GF7013 package to PYTHONPATH
GF7013_path = os.path.abspath(os.path.join(this_module_folder, '../../../..'))
if GF7013_path not in sys.path:
    sys.path.append(GF7013_path)

from GF7013.model_parameters import ensemble
from GF7013.sampling.metropolis_in_parallel import metropolis_in_parallel_POOL
from GF7013.sampling.metropolis import proposal_normal
from GF7013.probability_functions.pdf import pdf_uniform_nD

import numpy as np
import matplotlib.pyplot as plt

# define the pdf to sample (must have the likelihood/log_likelihood function defined)
class pdf_bimodal:
    def __init__(self, x_0, sigma_0, p_0, x_1, sigma_1, p_1):
        self.args_f0 = (x_0, sigma_0, p_0)
        self.args_f1 = (x_1, sigma_1, p_1)

    def __f(self, x, mean, sigma, p):
        alpha = 1 / sigma / np.sqrt(2 * np.pi)
        return alpha * np.exp(-0.5 * np.abs((x - mean) / sigma) ** p)

    def likelihood(self, x):
        return self.__f(x, *self.args_f0) + self.__f(x, *self.args_f1)

    def log_likelihood(self, x):
        like = self.likelihood(x)
        return -1e10 if like <= 0 else np.log(like)  # evitar log(0)


if __name__ == '__main__':
    # Parámetros bimodales
    x_0 = 0.5
    sigma_0 = 0.2
    p_0 = 2
    x_1 = 11
    sigma_1 = 0.0335
    p_1 = 1

    f = pdf_bimodal(x_0, sigma_0, p_0, x_1, sigma_1, p_1)

    # Dominio para evaluar pdf
    x_min = -15
    x_max = 22
    Num_x = 10000
    x_eval = np.linspace(x_min, x_max, Num_x)
    f_values = np.array([f.likelihood(x) for x in x_eval])
    dx = x_eval[1] - x_eval[0]
    f_area = np.sum(f_values) * dx

    # Parámetros prior uniforme
    lower = np.array([-10.0])
    upper = np.array([13.0])
    prior_pars = {"lower_lim": lower, "upper_lim": upper}
    prior = pdf_uniform_nD(par=prior_pars)

    # Proposal normal
    sigma_prop = (upper[0] - lower[0]) / 30
    cov = np.array([[sigma_prop ** 2]])
    proposal_pdf = proposal_normal(cov=cov)

    # Ensemble inicial con Nchains modelos distribuidos uniformemente
    Nchains = 1000
    rng = np.random.default_rng(seed=42)
    num_MCMC_steps = 1000
    use_log_likelihood = True
    beta = 1

    m0 = ensemble(Npar=1, Nmodels=Nchains, use_log_likelihood=use_log_likelihood, beta=beta)

    # Inicializar modelos aleatorios uniformes
    for i in range(Nchains):
        m0.m_set[i, :] = rng.uniform(low=lower[0], high=upper[0], size=1)

    # Corre el MCMC paralelo
    m, acceptance_ratios = metropolis_in_parallel_POOL(
        m0=m0,
        likelihood_fun=f,
        pdf_prior=prior,
        proposal=proposal_pdf,
        num_MCMC_steps=num_MCMC_steps,
        use_log_likelihood=use_log_likelihood
    )

    # Graficar resultados
    f_values_beta = np.array([f.likelihood(x) ** beta for x in x_eval])
    f_area_beta = np.sum(f_values_beta) * dx

    fig = plt.figure(figsize=(8, 10))

    # Primer subplot: PDF + histograma
    ax1 = fig.add_subplot(211)
    ax1.hist(m.m_set.flatten(), density=True, bins=50, color='red', alpha=1)

    ax1.plot(x_eval, f_values / f_area, label='Bimodal PDF', color='cyan')
    ax1.plot(x_eval, f_values_beta / f_area_beta, '--k', label=fr'Bimodal PDF $^{{\beta={beta}}}$')
    ax1.legend()
    ax1.set_ylabel('Probability Density')
    # Segundo subplot: caminata con línea continua y transparencia
    ax2 = fig.add_subplot(212, sharex=ax1)
    param_values = m.m_set.flatten()
    steps = np.arange(len(param_values))

    # Línea negra
    ax2.plot(param_values, steps, color='black', alpha=0.5, linewidth=.1)

    # Puntos rojos
    ax2.plot(param_values, steps, 'o', color='red', markersize=1, alpha=0.9)

    # Etiquetas
    ax2.set_xlabel("Parameter value")
    ax2.set_ylabel("Chain index (final parameter sample)")

    plt.show()

