import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from GF7013.models.ajuste_ortogonal_recta.recta import calc_xy_pred
from datos import obtener_datos_elipses

# importar modulos relevantes del paquete GF7013
pythonpackagesfolder = '../../../..' # no modificar si no se mueve de carpeta este notebook
import sys
sys.path.append(pythonpackagesfolder)

# modelo directo de la recta
from GF7013.models.ajuste_ortogonal_recta.forward import forward
from GF7013.model_parameters import ensemble
from GF7013.sampling.metropolis_in_parallel import metropolis_in_parallel_POOL
from GF7013.sampling.metropolis import proposal_normal
from GF7013.probability_functions.pdf import pdf_uniform_nD, pdf_normal
from GF7013.probability_functions.likelihood.likelihood_function import likelihood_function
from GF7013.models.ajuste_ortogonal_recta import recta

### MAIN CODE OF THE EXAMPLE.
if __name__ == '__main__':

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

    norm_dobs = np.sqrt((x_obs**2+y_obs**2))
    #
    min_a = -15
    max_a = 15

    min_theta = -360
    max_theta = 360

    par = {}
    par['lower_lim'] = np.array([min_a, min_theta])
    par['upper_lim'] = np.array([max_a, max_theta])
    LogOfZero = None
    rng = np.random.default_rng(888)
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

    # Modelos iniciales aleatorios provenientes de fprior
    Npar = int(len(fprior.par['lower_lim']))  # número de parámetros del modelo
    Nmodelos = int(1e4)
    modelos_iniciales = fprior._draw(Ns = Nmodelos)

    a_samples = modelos_iniciales[0]
    theta_samples = modelos_iniciales[1]

    modelos_iniciales_reshape = np.zeros((Nmodelos, Npar))
    for i in range(Nmodelos):
        modelos_iniciales_reshape[i, :] = [a_samples[i], theta_samples[i]]

    # Ensamble de modelos iniciales
    m0 = ensemble(Npar, Nmodelos, use_log_likelihood=True)
    m0.m_set = modelos_iniciales_reshape  # set the initial models in the ensemble

    # Número de pasos MCMC
    num_MCMC_steps = int(1E2)
    use_log_likelihood = True

    # Se utilizará el algortimo de Metropolis en paralelo en paralelo
    # para resolver el problema de ajuste de la recta a los datos
    m, acceptance_ratios = metropolis_in_parallel_POOL(m0 = m0,
               likelihood_fun = f, 
               pdf_prior = fprior,
               proposal = proposal_pdf, 
               num_MCMC_steps = num_MCMC_steps,
               use_log_likelihood = use_log_likelihood)
    

    # Graficando las muestras obtenidas del algoritmo de Metropolis


    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    # 1) Caminata con línea y puntos coloreados por log likelihood
    axs[0, 0].plot(m.m_set[:, 0], m.m_set[:, 1], 'k-', lw=0.05, alpha=0.3, label='Sample Path')
    sc = axs[0, 0].scatter(m.m_set[:, 0], m.m_set[:, 1], s=5, c=m.f, cmap='jet', label='Samples')
    axs[0, 0].set(xlabel='a', ylabel=r'$\theta$', title='Muestras Metropolis')
    axs[0, 0].grid(True)
    fig.colorbar(sc, ax=axs[0, 0], label='Log Likelihood')
    axs[0, 0].legend(loc='lower left')

    # 2) Histograma 2D
    H, xedges, yedges = np.histogram2d(m.m_set[:, 0], m.m_set[:, 1], bins=50)
    # Pcolormesh usa los bordes de los bins para hacer la malla
    X, Y = np.meshgrid(xedges, yedges)
    # Graficar con pcolormesh
    pcm = axs[0, 1].pcolormesh(X, Y, H.T, cmap='Blues', shading='auto', alpha=1)
    #
    axs[0, 1].set(xlabel='a', ylabel=r'$\theta$ (grados)', title='Mapa de densidad 2D')
    fig.colorbar(pcm, ax=axs[0, 1], label='Frecuencia')

    # 3) Histograma univariado de 'a'
    axs[0, 2].hist(m.m_set[:, 0], bins=50, alpha=0.7, color='blue')
    axs[0, 2].set(xlabel='a', ylabel='N° muestras', title='Histograma de a')

    # 4) Histograma univariado de 'theta'
    axs[1, 0].hist(m.m_set[:, 1], bins=50, alpha=0.7, color='green')
    axs[1, 0].set(xlabel=r'$\theta$ (grados)', ylabel='N° muestras', title='Histograma de theta')

    # 5) Ajuste recta y líneas de distancia coloreadas
    max_index = np.argmax(m.f)
    a, theta = m.m_set[max_index]
    m_plot = np.array([a, theta])
    axs[1, 1].errorbar(x_obs, y_obs, xerr=sigma_x, yerr=sigma_y, fmt='.r', capsize=2)

    recta.plot_recta(axs[1, 1], *m_plot, x_obs=x_obs, y_obs=y_obs, color='b', color_dist=None)
    x_pred, y_pred, _ = calc_xy_pred(*m_plot, x_obs, y_obs)
    distancias = np.linalg.norm(np.column_stack((x_pred - x_obs, y_pred - y_obs)), axis=1)
    segments = [[(x_pred[i], y_pred[i]), (x_obs[i], y_obs[i])] for i in range(len(x_pred))]
    lc = LineCollection(segments, cmap='jet', linewidths=2, alpha=1)
    lc.set_array(distancias)
    axs[1, 1].add_collection(lc)
    axs[1, 1].set_title('Ajuste recta y datos')
    cbar = fig.colorbar(lc, ax=axs[1, 1])
    cbar.set_label('Distancia')
    axs[1, 1].set_xlabel('X')
    axs[1, 1].set_ylabel('Y')
    axs[1, 1].grid(True)
    # 6) Eliminar último subplot sobrante
    fig.delaxes(axs[1, 2])

    plt.tight_layout()
    plt.savefig('resumen_metropolis.png', dpi=300)
    plt.show()