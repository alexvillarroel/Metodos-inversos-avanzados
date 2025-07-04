import matplotlib.pyplot as plt
import numpy as np

from datos import obtener_datos_elipses

# importar modulos relevantes del paquete GF7013
pythonpackagesfolder = '../../../..' # no modificar si no se mueve de carpeta este notebook
import sys
sys.path.append(pythonpackagesfolder)

# modelo directo de la recta
from GF7013.models.ajuste_ortogonal_recta.forward import forward
from GF7013.sampling.metropolis import metropolis
from GF7013.sampling.metropolis import proposal_normal
from GF7013.probability_functions.pdf import pdf_uniform_nD, pdf_normal
from GF7013.probability_functions.likelihood.likelihood_function import likelihood_function
from GF7013.models.ajuste_ortogonal_recta import recta


N = 25
semi_eje_mayor = 8
semi_eje_menor = 2
alpha = 45
delta_x = 0
delta_y = 4
desviacion_estandar_x = 1.0
desviacion_estandar_y = 1.0

x_obs, y_obs, sigma_x, sigma_y = obtener_datos_elipses(
                                        N = N,
                                        a = semi_eje_mayor,
                                        b = semi_eje_menor,
                                        alpha = alpha,
                                        deltax = delta_x,
                                        deltay = delta_y,
                                        sigma_x = desviacion_estandar_x,
                                        sigma_y = desviacion_estandar_y)

# Usando el algoritmo de Metropolis para ajustar la recta a los datos observados

# f prior
min_a = -15
max_a = 15

min_theta = -360
max_theta = 360

par = {}
par['lower_lim'] = np.array([min_a, min_theta])
par['upper_lim'] = np.array([max_a, max_theta])
LogOfZero = None
rng = np.random.default_rng(666)
fprior = pdf_uniform_nD(par = par, LogOfZero=LogOfZero, rng=rng)

# Modelo forward
forward_model = forward(x_obs, y_obs, sigma_x, sigma_y)

# Likelihood function
mu = np.zeros(N)
cov = np.eye(N) # matriz identidad
par = {'mu': mu, 'cov': cov}

f = likelihood_function(forward_model, pdf_normal(par=par, rng=rng))

# Distribution de proposición
sigma_prop_a = (max_a - min_a) / 100
sigma_prop_theta = (max_theta - min_theta) / 100
proposal_params = {}
proposal_params['cov'] = np.array([[sigma_prop_a, 0], [0, sigma_prop_theta]])  # matriz de covarianza 2D
proposal_pdf = proposal_normal(proposal_params['cov'])

# Modelo inicial
m0 = np.array([-10, -180])

NumSamples = int(1E5)
NumBurnIn =  0 # int(0.1 * NumSamples)
use_log_likelihood = True
beta = 1 

results = metropolis(m0= m0, 
                        likelihood_fun = f, 
                        pdf_prior = fprior, 
                        proposal = proposal_pdf, 
                        num_samples = NumSamples,
                        num_burnin = NumBurnIn,
                        use_log_likelihood = use_log_likelihood,
                        save_samples = True,
                        beta = beta)

# Graficando las muestras obtenidas del algoritmo de Metropolis
samples = results['samples']

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(samples.m_set[:, 0], samples.m_set[:, 1], 'r-', linewidth=0.1, label='Sample Path')
ax.scatter(samples.m_set[:, 0], samples.m_set[:, 1], s=1, c = samples.f, cmap='viridis', label='Samples')
ax.set_xlabel('a')
ax.set_ylabel('theta (grados)')
ax.set_title('Muestras del Algoritmo de Metropolis')
ax.grid(True)   
plt.colorbar(ax.collections[0], ax=ax, label='Log Likelihood')
plt.legend()
plt.tight_layout()
plt.savefig('metropolis_samples.png', dpi=300)
plt.show()

# Graficar las muestras
fig, ax = plt.subplots(2, 2, figsize=(8, 5))
ax[0, 0].hist2d(samples.m_set[:, 0], samples.m_set[:, 1], bins=50, cmap='Blues', alpha=0.7)
ax[0, 0].set_xlabel('a')
ax[0, 0].set_ylabel('theta (grados)')
ax[0, 0].set_title('Histograma 2D de Muestras de Metropolis')
ax[0, 1].hist(samples.m_set[:, 0], bins=50, alpha=0.7, color='blue')
ax[0, 1].set_xlabel('a')
ax[0, 1].set_ylabel('Número de muestras')
ax[0, 1].set_title('Histograma de muestras de a')
ax[1, 0].hist(samples.m_set[:, 1], bins=50, alpha=0.7, color='green')
ax[1, 0].set_title('Histograma de muestras de theta')
ax[1, 0].set_xlabel('theta (grados)')
ax[1, 0].set_ylabel('Número de muestras')
fig.delaxes(ax[1, 1])
plt.tight_layout()
plt.savefig('muestras_metropolis.png', dpi=300)
plt.show()



# Graficando la recta obtenida del ajuste a los datos observados
a = results["m"][0]
theta = results["m"][1]
m_plot = np.array([a, theta])

fig = plt.figure(1)
fig.clear()
fig.set_size_inches((6, 4))
ax = fig.add_subplot(111)
recta.plot_recta(ax, *m_plot, x_obs = x_obs, y_obs = y_obs, color_dist='c', color='b') 
ax.errorbar(x = x_obs, y=y_obs, xerr=sigma_x, yerr=sigma_y, fmt='.r', capsize = 2)
ax.axis('equal')
ax.grid('on')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Modelo de recta ajustado a los datos observados')
plt.savefig('ajuste_recta.png', dpi=300)
plt.show()