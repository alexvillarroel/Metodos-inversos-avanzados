import sys, os
# get folder of this module
this_module_folder = os.path.dirname(os.path.abspath(__file__))
# add GF7013 package to PYTHONPATH
GF7013_path = os.path.abspath(os.path.join(this_module_folder, '../../../..'))
sys.path.append(GF7013_path)
import numpy as np
import matplotlib.pyplot as plt

# === Importar funciones desde la estructura GF7013 ===
from GF7013.sampling.metropolis_in_parallel.metropolis_in_parallel_SERIAL import metropolis_in_parallel_SERIAL
from GF7013.model_parameters.ensemble import ensemble
from GF7013.sampling.metropolis.proposal_normal import proposal_normal
from GF7013.probability_functions.pdf.pdf_uniform_nD import pdf_uniform_nD

# === Definición de PDF bimodal ===
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
        return -1e10 if like <= 0 else np.log(like) # para evitar log(0)

# === Crear PDF bimodal sintética ===
f = pdf_bimodal(x_0=0.5, sigma_0=0.2, p_0=2,
                x_1=11, sigma_1=0.0335, p_1=1)

# === Prior uniforme entre -10 y 13 ===
lower = np.array([-10.0])
upper = np.array([13.0])
prior_pars = {"lower_lim": lower, "upper_lim": upper}
prior = pdf_uniform_nD(par=prior_pars)

# === Proposal normal ===
sigma_prop = (upper[0] - lower[0]) / 30
cov = np.array([[sigma_prop ** 2]])
proposal = proposal_normal(cov=cov)

# === Ensemble inicial ===
Nchains = 1000
rng = np.random.default_rng(seed=42)

m0 = ensemble(Npar=1, Nmodels=Nchains, use_log_likelihood=True, beta=1)
for i in range(Nchains): # hacemos el mismo numero de modelos que el largo de las cadenas
    m0.m_set[i, :] = rng.uniform(low=lower[0], high=upper[0], size=1)
# === Ejecutar Metropolis en paralelo (serial version) ===
m, tasas = metropolis_in_parallel_SERIAL(
    m0=m0,
    likelihood_fun=f,
    pdf_prior=prior,
    proposal=proposal,
    num_MCMC_steps=1000,
    num_burnin=0,  # No burn-in en este caso
    use_log_likelihood=True
)

# === Visualización de los modelos finales ===
plt.figure(figsize=(8, 4))
plt.hist(m.m_set[:, 0], bins=100, density=True, color='red', alpha=0.6, label='Muestras finales')
x = np.linspace(-10, 13, 1000)
f_vals = np.array([f.likelihood(xi) for xi in x])
f_vals /= np.trapz(f_vals, x)
plt.plot(x, f_vals, 'k--', label='PDF bimodal normalizada')
plt.title('Histograma de modelos finales - Metropolis en paralelo (serial)')
plt.xlabel('x')
plt.ylabel('Densidad')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Mostrar tasas
print("Tasa de aceptación promedio:", np.mean(tasas))