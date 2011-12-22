#cython: embedsignature=True
#cython: cdivision=True
#cython: wraparound=False
#cython: boundscheck=False

from copy import copy
import numpy as np
cimport numpy as np

from scipy.stats import norm
from scipy.integrate import quad

cimport cython

from libc.math cimport log, exp, sqrt, pow

include "gsl/gsl.pxi"

ctypedef double * double_ptr
ctypedef void * void_ptr

cdef extern from "stdlib.h":
    void free(void* ptr)
    void* malloc(size_t size)
    void* realloc(void* ptr, size_t size)

    
cdef inline double ExGauss_pdf(double value, double mu, double sigma, double tau, bint logp=True):
    """ ExGaussian log pdf"""
    cdef double z
    z = value - mu - sigma*sigma/tau
    return -log(tau) - (z+(sigma*sigma/(2*tau)))/tau + log( gsl_cdf_gaussian_P(z/sigma, 1) )

cdef inline double ExGauss_cdf(double value, double mu, double sigma, double tau, bint logp=True):
    """ ExGaussian log cdf upper tail"""
    return log((1-(gsl_cdf_gaussian_P(((value-mu)/sigma), 1) - exp(((sigma*sigma)/(2*(tau*tau)))-((value-mu)/tau) ) * gsl_cdf_gaussian_P(((value-mu)/sigma)-(sigma/tau), 1))))

def Go(np.ndarray[double, ndim=1] value, double imu_go, double isigma_go, double itau_go):
    """Ex-Gaussian log-likelihood of GoRTs"""
    assert imu_go > 0
    assert isigma_go > 0
    assert itau_go > 0
    assert np.all(value != -999.)

    cdef Py_ssize_t size = value.shape[0]
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0

    for i in range(size):
        p = ExGauss_pdf(value[i], imu_go, isigma_go, itau_go)
               
        if np.isinf(p) or np.isnan(p):
            return -np.inf
        
        sum_logp += p

    return sum_logp

def SRRT(np.ndarray[double, ndim=1] value, np.ndarray[int, ndim=1] issd, double imu_go, double isigma_go, double itau_go, double imu_stop, double isigma_stop, double itau_stop):
    """Censored ExGaussian log-likelihood of SRRTs"""
    assert imu_go > 0
    assert isigma_go > 0
    assert itau_go > 0
    assert imu_stop > 0
    assert isigma_stop > 0
    assert itau_stop > 0
    assert np.all(value != -999.)

    cdef Py_ssize_t size = value.shape[0]
    cdef Py_ssize_t i
    cdef double p
    cdef double sum_logp = 0

    for i in range(size):
        p = ExGauss_pdf(value[i], imu_go, isigma_go, itau_go) + ExGauss_cdf(value[i], imu_stop+issd[i], isigma_stop, itau_stop)

        if np.isinf(p) or np.isnan(p):
            return -np.inf
        
        sum_logp += p

    return sum_logp


def CExGauss_I(double value, double imu_go, double isigma_go, double itau_go, double imu_stop, double isigma_stop, double itau_stop, int issd):
    cdef double pdf
    pdf = exp(ExGauss_cdf(value, imu_go, isigma_go, itau_go)) * exp(ExGauss_pdf(value, imu_stop+issd, isigma_stop, itau_stop))

    if np.isinf(pdf) or np.isnan(pdf):
       return -np.inf
    else:
       return pdf
        
def Inhibitions(np.ndarray[int, ndim=2] value, double imu_go, double isigma_go, double itau_go, double imu_stop, double isigma_stop, double itau_stop):
    """Censored ExGaussian log-likelihood of inhibitions"""
    assert imu_go > 0
    assert isigma_go > 0
    assert itau_go > 0
    assert imu_stop > 0
    assert isigma_stop > 0
    assert itau_stop > 0
    
    cdef Py_ssize_t size = value.shape[0]
    cdef Py_ssize_t i
    cdef double p
    
    cdef double sum_logp = 0
    cdef double p_ssd

    cdef int ssd, n_trials

    for i in range(size):
        ssd = value[i, 0]
        n_trials = value[i, 1]
        
        assert (ssd != -999)
        
        # Compute probability of single SSD
        p_ssd = log(quad(CExGauss_I, 0, 6000, args=(imu_go, isigma_go, itau_go, imu_stop, isigma_stop, itau_stop, ssd))[0])
        # Multiply with number of trials and add to overall p
        sum_logp += n_trials * p_ssd
        
    return sum_logp
