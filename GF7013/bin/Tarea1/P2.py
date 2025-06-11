# -*- python -*-
# -*- coding: utf-8 -*-
"""
Pregunta 2.2 y 2.3 – Tarea 1 GF7013
Código original: Nacho (modificado mínimamente)
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys

# Agregar GF7013 al PYTHONPATH
path = os.path.join('..', '..', '..')
path = os.path.abspath(path)
sys.path.append(path)

# Importar clase pdf_uniform_nD
from GF7013.probability_functions import pdf

# ---------- CONFIGURACIÓN ----------
Ns = int(1e5)  # Cambiar a 1e4 o 1e6 para P2.3

# Parámetros para la distribución uniforme en R^2
lower_lim = np.array([-2, 3.2])
upper_lim = np.array([5, 7.0])
par = {'lower_lim': lower_lim, 'upper_lim': upper_lim}

# Instanciar pdf uniforme
pdf_uniform = pdf.pdf_uniform_nD(par=par)

# Generar muestras
samples = pdf_uniform.draw(Ns)

# ---------- HISTOGRAMA 2D CONJUNTO ----------
fig = plt.figure(figsize=(10, 4))

# Histograma conjunto (2D)
ax1 = fig.add_subplot(1, 2, 1)
h = ax1.hist2d(samples[0], samples[1], bins=7, cmap='viridis')
ax1.set_title(f'Histograma conjunto (Ns = {Ns:.0e})')
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
plt.colorbar(h[3], ax=ax1, label='Frecuencia')

# ---------- HISTOGRAMAS MARGINALES ----------
ax2 = fig.add_subplot(2, 2, 2)
ax2.hist(samples[0], bins=50, color='blue', edgecolor='black')
ax2.set_title('Histograma marginal de X1')
ax2.set_xlabel('X1')

ax3 = fig.add_subplot(2, 2, 4)
ax3.hist(samples[1], bins=50, color='green', edgecolor='black')
ax3.set_title('Histograma marginal de X2')
ax3.set_xlabel('X2')

plt.tight_layout()
plt.show()
