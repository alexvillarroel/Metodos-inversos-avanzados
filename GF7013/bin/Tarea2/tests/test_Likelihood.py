import numpy as np
import matplotlib.pyplot as plt

# importar modulos relevantes del paquete GF7013
pythonpackagesfolder = '../../../..' # no modificar si no se mueve de carpeta este notebook
import sys
sys.path.append(pythonpackagesfolder)

from GF7013.bin.Tarea2.P1.datos import obtener_datos_elipses
from GF7013.model_parameters.ensemble import ensemble
from GF7013.probability_functions.pdf import pdf_uniform_nD,pdf_normal
from GF7013.probability_functions.likelihood.likelihood_function import likelihood_function
from GF7013.models.ajuste_ortogonal_recta.forward import forward


# Generar datos sinteticos
N = int(70)
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

# definición de los parámetros
norm_dobs = np.sqrt((x_obs**2+y_obs**2))
#
ll_a, ul_a = -2*np.max(norm_dobs), 2* np.max(norm_dobs)
ll_theta, ul_theta = -180,180 
#
lower_lim = np.array([ll_a,ll_theta])
upper_lim = np.array([ul_a,ul_theta])

# Valores de la grilla

theta_values = np.linspace(ll_theta, ul_theta, 100)
a_values = np.linspace(ll_a, ul_a, 100)

Nmodels = len(theta_values) * len(a_values)
Npar=2

a_grid,theta_grid = np.meshgrid(a_values,theta_values, indexing='ij')
m_posible_values = np.column_stack([a_grid.ravel(),theta_grid.ravel()])


# Forward model
forward_model = forward(x_obs, y_obs, sigma_x, sigma_y)
from GF7013.models.ajuste_ortogonal_recta.recta import calc_dist_sigma
dist=np.zeros((len(m_posible_values),N))
deltas = np.zeros((len(m_posible_values),N))

for i,m_value in enumerate(m_posible_values):
    dist[i,:],deltas[i,:],_,_,_ =calc_dist_sigma(m_value, x_obs, y_obs, sigma_x, sigma_y)

print(np.shape(dist),np.shape(deltas))

# f_priori

# likelihood
LogOfZero = None
rng = np.random.default_rng(66)
rng_uniform = np.random.default_rng(77)
# par
likelihood=np.zeros(Nmodels)
#
likelihood_f_prior=np.zeros(Nmodels)
# a posteriori
likelihood_f_posterior= np.zeros(Nmodels)
#### log_likelihood
# par
loglikelihood=np.zeros(Nmodels)
#
loglikelihood_f_prior=np.zeros(Nmodels)
# a posteriori
loglikelihood_f_posterior= np.zeros(Nmodels)

for i,m_value in enumerate(m_posible_values):
    mu = np.zeros(N)
    x = dist[i,:]
    cov = np.diag(deltas[i,:])
    par = {'mu': mu, 'cov': cov}
    par_uniform = {'lower_lim':lower_lim,'upper_lim':upper_lim}
    likelihood_f_prior[i] = pdf_uniform_nD(par = par_uniform,rng=rng_uniform)._likelihood(m_value)
    likelihood[i] = pdf_normal(par,rng=rng)._likelihood(x)
    likelihood_f_posterior[i] = likelihood_f_prior[i] * likelihood[i]
    #
    loglikelihood_f_prior[i] = pdf_uniform_nD(par = par_uniform,rng=rng_uniform)._log_likelihood(m_value)
    loglikelihood[i] = pdf_normal(par,rng=rng)._log_likelihood(x)
    loglikelihood_f_posterior[i] = loglikelihood_f_prior[i] + loglikelihood[i]



ensamble = ensemble(Npar=Npar,Nmodels=Nmodels,use_log_likelihood=False)

ensamble.fprior=likelihood_f_prior
ensamble.like=likelihood
ensamble.f = likelihood_f_posterior

ensamble_log = ensemble(Npar=Npar,Nmodels=Nmodels,use_log_likelihood=True)

ensamble_log.fprior=loglikelihood_f_prior
ensamble_log.like=loglikelihood
ensamble_log.f = loglikelihood_f_posterior



# plot
# fig,axs= plt.subplots(nrows=3,ncols=2)
# axs[0,0].pcolor(theta_grid, a_grid,ensamble.fprior)
# axs[0,0].set_title('Prior Log Likelihood')
# axs[0,1].pcolor(theta_grid, a_grid,ensamble_log.fprior)
# axs[0,1].set_title('Prior Likelihood')
# axs[1,0].pcolor(theta_grid, a_grid,ensamble.like)
# axs[1,0].set_title('Likelihood')
# axs[1,1].pcolor(theta_grid, a_grid,ensamble_log.like)
# axs[1,1].set_title('Log Likelihood')
# axs[2,0].pcolor(theta_grid, a_grid,ensamble.f)
# axs[2,0].set_title('Posterior Likelihood')
# axs[2,1].pcolor(theta_grid, a_grid,ensamble_log.f)
# axs[2,1].set_title('Posterior Log Likelihood')
# plt.tight_layout()
# plt.show()

# scatter
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 6))
axs[0, 0].scatter(m_posible_values[:, 0], m_posible_values[:, 1], c=ensamble.fprior, cmap='viridis', s=10)
axs[0, 0].set_title('Prior Log Likelihood')
axs[0, 1].scatter(m_posible_values[:, 0], m_posible_values[:, 1], c=ensamble_log.fprior, cmap='viridis', s=10)
axs[0, 1].set_title('Prior Likelihood')
axs[1, 0].scatter(m_posible_values[:, 0], m_posible_values[:, 1], c=ensamble.like, cmap='viridis', s=10)
axs[1, 0].set_title('Likelihood')
axs[1, 1].scatter(m_posible_values[:, 0], m_posible_values[:, 1], c=ensamble_log.like, cmap='viridis', s=10)
axs[2, 0].scatter(m_posible_values[:, 0], m_posible_values[:, 1], c=ensamble.f, cmap='viridis', s=10)
axs[2, 0].set_title('Posterior Likelihood')
axs[2, 1].scatter(m_posible_values[:, 0], m_posible_values[:, 1], c=ensamble_log.f, cmap='viridis', s=10)
axs[2, 1].set_title('Posterior Log Likelihood')
plt.tight_layout()
plt.show()