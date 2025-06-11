import numpy as np
import matplotlib.pyplot as plt

# importar modulos relevantes del paquete GF7013
pythonpackagesfolder = '../../../..' # no modificar si no se mueve de carpeta este notebook
import sys
sys.path.append(pythonpackagesfolder)

from GF7013.bin.Tarea2.P1.datos import obtener_datos_elipses
from GF7013.model_parameters.ensemble import ensemble
from GF7013.probability_functions.pdf import pdf_uniform_nD
from GF7013.probability_functions.likelihood.likelihood_function import likelihood_function
from GF7013.models.ajuste_ortogonal_recta.forward import forward


# Generar datos sinteticos
N = 70
semi_eje_mayor = 10
semi_eje_menor = 2
alpha = -45
delta_x = 0
delta_y = -10
desviacion_estandar_x = 1.0
desviacion_estandar_y = 0.5

x_obs, y_obs, sigma_x, sigma_y = obtener_datos_elipses(
                                        N = N,
                                        a = semi_eje_mayor,
                                        b = semi_eje_menor,
                                        alpha = alpha,
                                        deltax = delta_x,
                                        deltay = delta_y,
                                        sigma_x = desviacion_estandar_x,
                                        sigma_y = desviacion_estandar_y)

# Valores de la grilla
min_theta = -180
max_theta = 180

min_a = - 2 * 18.51
max_a =   2 * 18.51

theta_values = np.linspace(min_theta, max_theta, 100)
a_values = np.linspace(min_a, max_a, 100)

Nmodels = len(theta_values) * len(a_values)
Npar = 2

grid = np.zeros((Nmodels, Npar))
for i, theta in enumerate(theta_values):
    for j, a in enumerate(a_values):
        grid[i * len(a_values) + j, 0] = theta
        grid[i * len(a_values) + j, 1] = a

# f prior
par = {}
par["lower_lim"] = np.array([min_a, min_theta])
par["upper_lim"] = np.array([max_a, max_theta])
LogOfZero = None
rng = np.random.default_rng(66)
f_prior = pdf_uniform_nD(par, LogOfZero=LogOfZero, rng=rng)

# Forward model
forward_model = forward(x_obs, y_obs, sigma_x, sigma_y)

# Likelihood function
likelihood_function = likelihood_function(forward_model, f_prior)

# Ensemble
ensemble = ensemble(Npar, Nmodels, use_log_likelihood=True)
ensemble.m_set = grid
ensemble.fprior = likelihood_function.likelihood(ensemble.m_set)
ensemble.like = 

