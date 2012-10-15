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
from cython.parallel import *

from libc.math cimport log, exp, sqrt, pow

cdef extern from "math.h":
    double INFINITY, NAN

from cython_gsl cimport *

ctypedef double * double_ptr
ctypedef void * void_ptr

cdef extern from "stdlib.h":
    void free(void* ptr)
    void* malloc(size_t size)
    void* realloc(void* ptr, size_t size)


cdef inline double ExGauss_pdf(double value, double mu, double sigma, double tau) nogil:
    """ ExGaussian log pdf"""
    cdef double z
    if tau > 0.05*sigma:
        z = value - mu - ((sigma**2)/tau)
        return -log(tau) - (z+(sigma**2/(2*tau)))/tau + log( gsl_cdf_gaussian_P(z/sigma, 1) )
    else: 
        return log(gsl_ran_gaussian_pdf(value-mu, sigma))

cdef inline double ExGauss_cdf(double value, double mu, double sigma, double tau) nogil:
    """ ExGaussian log cdf upper tail"""
    cdef double z
    if tau > 0.05*sigma:
        z = value - mu - ((sigma**2)/tau)
        return log(1-(gsl_cdf_gaussian_P((value-mu)/sigma,1)-gsl_cdf_gaussian_P(z/sigma,1)*exp((((mu+((sigma**2)/tau))**2)-
        (mu**2)-2*value *((sigma**2)/tau))/(2*(sigma**2)))))   
    else:
        return log((1-(gsl_cdf_gaussian_P(((value-mu)/sigma), 1))))

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
        p_ssd = log(integrate_cexgauss(0, 6000, imu_go, isigma_go, itau_go, imu_stop, isigma_stop, itau_stop, ssd))
        if np.isinf(p_ssd) or np.isnan(p_ssd):
            return -np.inf
        # Multiply with number of trials and add to overall p
        sum_logp += n_trials * p_ssd

    return sum_logp

########################
# Integration routines #
########################
cdef double eval_cexgauss(double x, void * params) nogil:
    cdef double imu_go, isigma_go, itau_go, imu_stop, isigma_stop, itau_stop, issd
    cdef double p
    imu_go = (<double_ptr> params)[0]
    isigma_go = (<double_ptr> params)[1]
    itau_go = (<double_ptr> params)[2]
    imu_stop = (<double_ptr> params)[3]
    isigma_stop = (<double_ptr> params)[4]
    itau_stop = (<double_ptr> params)[5]
    issd = (<double_ptr> params)[6]

    p = exp(ExGauss_cdf(x, imu_go, isigma_go, itau_go)) * exp(ExGauss_pdf(x, imu_stop+issd, isigma_stop, itau_stop))

    return p

cdef double integrate_cexgauss(double lower, double upper, double imu_go, double isigma_go, double itau_go, double imu_stop, double isigma_stop, double itau_stop, int issd):

    cdef double alpha, result, error, expected
    cdef gsl_function F
    cdef double params[7]
    cdef size_t neval
    cdef gsl_integration_workspace * W
    W = gsl_integration_workspace_alloc(5000)

    params[0] = imu_go
    params[1] = isigma_go
    params[2] = itau_go
    params[3] = imu_stop
    params[4] = isigma_stop
    params[5] = itau_stop
    params[6] = issd

    F.function = &eval_cexgauss
    F.params = params

    gsl_integration_qag(&F, lower, upper, 1e-4, 1e-4, 5000, GSL_INTEG_GAUSS41, W, &result, &error)
    gsl_integration_workspace_free(W)

    return result
