import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../../..')))
from GF7013.probability_functions.pdf import pdf_uniform_nD
from GF7013.bin.Tarea2.P1.datos import obtener_datos_elipses
from GF7013.models.ajuste_ortogonal_recta import recta

N = 25
semi_eje_mayor = 8
semi_eje_menor = 2
alpha = 45
delta_x = 0
delta_y = 4
desviacion_estandar_x = 1.0
desviacion_estandar_y = 1.0
# valores de d
x_obs, y_obs, sigma_x, sigma_y = obtener_datos_elipses(
                                        N = N,
                                        a = semi_eje_mayor,
                                        b = semi_eje_menor,
                                        alpha = alpha,
                                        deltax = delta_x,
                                        deltay = delta_y,
                                        sigma_x = desviacion_estandar_x,
                                        sigma_y = desviacion_estandar_y)
# valores de m 
norm_dobs = np.sqrt((x_obs**2+y_obs**2))
#
ll_a, ul_a = -2*np.max(norm_dobs), 2* np.max(norm_dobs)
ll_theta, ul_theta = -180,180 
#
lower_lim = np.array([ll_a,ll_theta])
upper_lim = np.array([ul_a,ul_theta])
par = {'lower_lim': lower_lim,'upper_lim':upper_lim}
pdf_uniforme = pdf_uniform_nD(par)
#
N_samples = 1e5
#
LogOfZero = None
rng = np.random.default_rng(42)  # Para reproducibilidad
f_prior = pdf_uniform_nD(par, LogOfZero=LogOfZero, rng=rng)
samples = f_prior._draw(N_samples)
def histogram_p2_1(samples,vmin,vmax):
    """
    Do a figure with the hist2d and marginals of samples, assuming a dimension of samples 
    samples.shape = 2,~
    vmin: minimum value of colorbar in the hist2d
    vmax: maximum value of colorbar in the hist2d
    """
    if samples.shape[0] == 2:
        # Crear figura y grilla
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 2], height_ratios=[1, 1], wspace=0.4, hspace=0.4)

        # Histograma 2D
        ax_main = fig.add_subplot(gs[0:2, 0:2])  # Ocupa las posiciones (0,0), (0,1), (1,0), (1,1)
        hist = ax_main.hist2d(samples[0, :], samples[1, :], bins=40, vmin=vmin, vmax=vmax)

        ax_main.set_xlabel('a parameter', fontsize=12)
        ax_main.set_ylabel(r'$\theta$ parameter', fontsize=12)
        ax_main.set_title('Histogram of Samples from Uniform PDF')
        ax_main.grid()

        # Colorbar
        cb = fig.colorbar(hist[3], ax=ax_main)
        cb.set_label('Counts')

        # Histograma marginal de samples[0,:] (eje X)
        ax_marg_x = fig.add_subplot(gs[0, 2])
        ax_marg_x.hist(samples[0, :], bins=40, orientation='vertical', color='green', edgecolor='black')
        ax_marg_x.set_title('Marginal of a parameter')
        ax_marg_x.set_xlabel('Counts')
        ax_marg_x.set_ylabel('a parameter')
        ax_marg_x.grid()

        # Histograma marginal de samples[1,:] (eje Y)
        ax_marg_y = fig.add_subplot(gs[1, 2])
        ax_marg_y.hist(samples[1, :], bins=40, orientation='vertical', color='blue', edgecolor='black')
        ax_marg_y.set_title(r'Marginal of $\theta$ parameter')
        ax_marg_y.set_xlabel('Counts')
        ax_marg_y.set_ylabel(r'$\theta$ parameter')
        ax_marg_y.grid()
    else:
        raiseError('The shape of samples has not 2 parameters')
    return fig
fig = histogram_p2_1(samples,0,100)
plt.show()

