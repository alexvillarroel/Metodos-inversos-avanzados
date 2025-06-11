import numpy as np
import matplotlib.pyplot as plt

pythonpackagesfolder = '../../../..' # no modificar si no se mueve de carpeta este notebook
import sys
sys.path.append(pythonpackagesfolder)

from GF7013.probability_functions.pdf import pdf_uniform_nD

N_samples = 1e5
par = {}
par["lower_lim"] = np.array([-2 * 18.51, -180])
par["upper_lim"] = np.array([ 2 * 18.51, 180])
LogOfZero = None
rng = np.random.default_rng(42)  # Para reproducibilidad

f_prior = pdf_uniform_nD(par, LogOfZero=LogOfZero, rng=rng)
samples = f_prior._draw(N_samples)

# Graficar el histograma de las muestras
plt.figure(figsize=(10, 6))
plt.hist2d(samples[0, :], samples[1, :], bins=80)
plt.xlabel('a')
plt.ylabel('theta')
plt.title('Histogram of Samples from Uniform PDF')
plt.grid()
plt.show()
