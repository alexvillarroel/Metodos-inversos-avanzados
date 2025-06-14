# -*- coding: utf-8 -*-
"""
Prof. Francisco Hernan Ortega Culaciati
ortega.francisco@uchile.cl
Departamento de Geofisica - FCFM
Universidad de Chile

Defines a class for a multivariate normal probability function.
It must provide 5 member functions, inherited from pdf_base

self.likelihood(x): gives the value of the probability function (unnormalized).
sef.log_likelihood(x): gives the value of the log of probability function (unnormalized).
self.pdf(x): gives the value of the normalized pdf
self.log_pdf(x): gives the log of the value of the normalized pdf.
self.draw(N = None): produces  realizations of the distribution. 
"""

from .pdf_base import pdf_base
import numpy as np


class pdf_normal(pdf_base):
    """
    Defines a class for a n-dimensional multivariate normal N(mu, cov) probability 
    function class.
    """
    ####
    def __init__(self, par, rng = None):
        """
        par: a dictionary containing the parameters that define the pdf.
        allowed keys are:
           'mu': a 1D array with the expected value (mean) of the normal pdf.
                 all values must be finite.
           'cov': a 2D array with the covariance matrix of the normal pdf.
                  must be a symmetric, nonsingular and positive definite matrix.
        rng: an instance of a numpy random number generator. If rng is None we use
            numpy.random.default_rng()
        """
        # The base class constructor assigns self.par = par
        super().__init__(par, rng)

        self.type = 'multivariate normal'

        
        # make relevant parameters easier to access
        self.mu = self.par['mu']
        self.cov = self.par['cov']
        self.N = len(self.mu)

        # compute inverse of covariance matrix needed for evaluation of likelihood/pdf
        self.inv_cov = np.linalg.inv(self.cov)

        # compute cholesky decomposition of self.cov for sample generation.
        self.right_chol_cov = np.linalg.cholesky(self.cov) # A.dot(A.T) = self.cov

        # compute normalization constant and base e logarithm of normalization.
        # (see self._pdf)
        sign, logdetCov = np.linalg.slogdet(self.cov)
        self.log_normalization = -0.5 * (self.N * np.log(2 * np.pi) + logdetCov)
        self.normalization = np.exp(self.log_normalization)

        if self.normalization < 1E3 * np.finfo(float).eps:
           print('Floating point overflow when calculating the '
                        +'normalization constant.')
           print('Use log_pdf or (log) Likelihood values instead of pdf.')
           self.normalization = None


    ####
    def __check_x(self, x):
        """
        Check that the array x has the correct shape and size
        array must be a single column 2D array or a 1D array
        """
        if x.ndim != 1:
            x = x.flatten()
            
        if len(x) != self.N:
           raise ValueError('x has size {:d}, must be a 1D array of length {:d}'.format(
                                                                          len(x), self.N))
        return x
        

    ####
    def _log_likelihood(self, x):
        """
        computes the base e logarithm of the likelihood of the vector value x. 
        x must be a numpy array of 1 D, or a column vector 2D numpy array.
        
        TODO: For the homework please compute directly the logarithm of the (unnormalized)
        likelihood ( DO NOT TAKE THE LOGARITHM OF THE PDF). You must program the formula.

        """
        x = self.__check_x(x)
        misfit = x - self.mu
        log_likelihood_value = -0.5 * np.dot(misfit.T, np.dot(self.inv_cov, misfit))
        return log_likelihood_value
        
    ####
    def _likelihood(self, x):
        """
        computes the (unnormalized) likelihood of the vector value x. 
        x must be a numpy array of 1 D.

        TODO: A hint: use what you already coded!. 
        
        """
        LogLike = self._log_likelihood(x)
        return np.exp(LogLike)

    
    ####
    def _log_pdf(self, x):
        """
        computes the value of the probability density function (i.e., normalized pdf) 
        for x.
        """
        return self._log_likelihood(x) + self.log_normalization
        
    ####
    def _pdf(self, x):
        """
        computes the value of the probability density function (i.e., normalized pdf) 
        for x.
        """
        return  self._likelihood(x) * self.normalization

    ####
    def _draw(self, Ns = None):
        """
        produces numpy array with Ns realizations of the probability distribution.
        If Ns is None, returns a 1D numpy array with the shape of self.mu .
        if Ns is a positive integer, returns a 2D numpy array where the number of rows
        is the length of self.mu and the number of columns is Ns (i.e., each column of the
        array is a sample of the multivariate normal pdf).

        HINT: Aquí puede usar el método relacional y el generador de números aleatorios 
        de python para la distribución Normal Estándar o Canónica N(0,1) dado por 
        la función de numpy rng.standard_normal().

        """
        if Ns is None:
            # Realization of a standard normal distribution
            u = self.rng.standard_normal(size=self.N)

            # Using functional relational form
            sample = self.mu + self.right_chol_cov @ u
        else:
            # Initialize the sample array
            sample = np.zeros((int(self.N), int(Ns)))

            # Generate samples
            for i in range(int(Ns)):
                # Generate a standard normal random variable
                u = self.rng.standard_normal(self.N)

                # Using functional relational form
                sample[:, i] = self.mu + np.dot(self.right_chol_cov, u)
        return sample

    def test_draw_samples(self, samples, tol_mu=0.05, tol_cov=0.1, verbose=True):
        """
        Evalúa el rendimiento de la función de muestreo, en base a 
        ciertas tolerancias para la media y la covarianza.
        samples: array de muestras, donde cada columna es una muestra
        tol_mu: tolerancia para la media
        tol_cov: tolerancia para la covarianza
        verbose: si True, imprime los resultados de la evaluación
        """

        # Se calcula la media y covarianza empírica de las muestras
        sample_mean = np.mean(samples, axis=1)
        sample_cov = np.cov(samples, rowvar=1)

        # Se calcula la diferencia entre la media y covarianza empírica y la teórica
        mean_diff = np.abs(sample_mean - self.mu)
        cov_diff = np.abs(sample_cov - self.cov)

        # Checkea si todas componentes de la media y covarianza están dentro de las tolerancias
        if np.all(mean_diff < tol_mu) and np.all(cov_diff < tol_cov):
            if verbose:
                print("Se pasó el test de muestreo.")
                print(f"Diferencia de media: {mean_diff}")
                print(f"Diferencia de covarianza:\n{cov_diff}")
            return True
        else:
            if verbose:
                print("No se pasó el test de muestreo.")
                print(f"Diferencia de media: {mean_diff}")
                print(f"Diferencia de covarianza:\n{cov_diff}")
            return False
