# -*- python -*-
# -*- coding: utf-8 -*-
"""
Francisco Hernán Ortega Culaciati
ortega.francisco@uchile.cl
Departamento de Geofísica - FCFM
Universidad de Chile


Make some testing of the multivariate normal distribution
Modifications: 

"""
import numpy as np
import sys, os 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

from scipy.stats import norm

# add GF7013 location to PYTHONPATH
path = os.path.join('..','..','..')
path = os.path.abspath(path)
sys.path.append(path)

# now import GF7013
import GF7013 
from GF7013.probability_functions import pdf

# ---------- FUNCIÓN PARA GRAFICAR ----------

def get_figure_results_normal(df,mu,cov,bins=50):
    """
    Función para graficar histogramas y distribuciones de probabilidad
    de muestras generadas a partir de una distribución normal 2D.
    Args:
        df (pd.DataFrame): DataFrame que contiene las muestras generadas.
        mu (array): Media de la distribución.
        cov (array): Matriz de covarianza.
    """
    if df.shape[1] != 2:
        raise ValueError("El DataFrame debe contener exactamente dos columnas.")
    
    # Crear figura con GridSpec para controlar el layout
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig)

    # ---------- Histograma conjunto 2D ----------
    ax1 = fig.add_subplot(gs[0, 0])
    x = df['X1']
    y = df['X2']
    Ns = len(x)

    h2d = ax1.hist2d(x, y, bins=bins, cmap='viridis',cmin=1)
    fig.colorbar(h2d[3], ax=ax1, label='Frecuencia')
    ax1.set_title(f'Histograma conjunto 2D Ns = {Ns:.0e}')
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')

    # ---------- Histograma conjunto 3D ----------
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)

    xpos, ypos = np.meshgrid(xedges[:-1] + np.diff(xedges)/2,
                             yedges[:-1] + np.diff(yedges)/2, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    dz = hist.ravel()

    mask = dz > 0
    xpos = xpos[mask]
    ypos = ypos[mask]
    zpos = zpos[mask]
    dz = dz[mask]
    dx = dy = (xedges[1] - xedges[0]) * np.ones_like(dz)

    ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', cmap='viridis')
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.set_zlabel('Frecuencia')
    ax2.set_title(f'Histograma conjunto 3D Ns = {Ns:.0e}')

    # ---------- Histogramas marginales ----------
    ax3 = fig.add_subplot(gs[1, :])  # Ocupa toda la fila inferior

    # Histograma X1
    ax3.hist(df['X1'], bins=50, density=True, color='skyblue', alpha=0.5,
            label='Marginal X1', edgecolor='skyblue', linewidth=1)

    # Histograma X2
    ax3.hist(df['X2'], bins=50, density=True, color='lightgreen', alpha=0.5,
            label='Marginal X2', edgecolor='lightgreen', linewidth=1)

    # FDP teóricas
    x1_line = np.linspace(df['X1'].min(), df['X1'].max(), 200)
    y1_line = norm.pdf(x1_line, loc=mu[0], scale=np.sqrt(cov[0, 0]))
    ax3.plot(x1_line, y1_line, color='blue', lw=2, label='FDP teórica X1')

    x2_line = np.linspace(df['X2'].min(), df['X2'].max(), 200)
    y2_line = norm.pdf(x2_line, loc=mu[1], scale=np.sqrt(cov[1, 1]))
    ax3.plot(x2_line, y2_line, color='green', lw=2, label='FDP teórica X2')

    ax3.set_title(f'Histogramas y FDP teóricas – X1 y X2 con Ns = {Ns:.0e}')
    ax3.set_xlabel('Valor')
    ax3.set_ylabel('Densidad')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()



# ---------- BLOQUE PRINCIPAL ----------
if __name__ == '__main__':
    # Definir parámetros
    mu = np.array([0.5, 3.0])
    cov = np.array([[2, 1], [1, 4]])
    Ns = int(1e5)

    par = {'mu': mu, 'cov': cov}

    # Instanciar la distribución normal multivariada
    normal = pdf.pdf_normal(par=par)

    # Generar muestras
    samples = normal.draw(Ns=Ns)  # shape: (2, Ns)

    # Histograma conjunto (X1 vs X2)
    df = pd.DataFrame(samples.T, columns=['X1', 'X2'])
    get_figure_results_normal(df, mu, cov)
    # Testeo: comparación con media y covarianza empírica
    normal.test_draw_samples(samples)
