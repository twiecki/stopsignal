#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import numpy.lib.recfunctions as rec
import pymc as pm
from copy import copy
from scipy.stats import norm
from scipy.integrate import quad
try:
    from IPython.Debugger import Tracer; debug_here = Tracer()
except:
    def debug_here(): pass

import kabuki
from kabuki.hierarchical import Parameter

import stop_likelihoods

def ExGauss_pdf(value, mu, sigma, tau):
    """ ExGaussian log pdf"""
    z = value - mu - sigma*sigma/tau
    return -np.log(tau)-(z+(sigma*sigma/(2*tau)))/tau+np.log(norm.cdf(z/sigma))

def ExGauss_cdf(value, mu, sigma, tau):
    """ ExGaussian log cdf upper tail"""
    exp_term = np.exp(((sigma**2)/(2*(tau**2)))-((value-mu)/tau))
    cdf_2 = norm.cdf(((value-mu)/sigma)-(sigma/tau))
    if sigma*.15 < tau or (exp_term == np.inf and cdf_2 == 0):
        return np.log((1-(norm.cdf(((value-mu)/sigma)))))
    else:
        return np.log((1-(norm.cdf(((value-mu)/sigma)) - exp_term * cdf_2)))

def cython_Go(value, imu_go, isigma_go, itau_go):
    """Ex-Gaussian log-likelihood of GoRTs"""
    return stop_likelihoods.Go(value, imu_go, isigma_go, itau_go)

def cython_SRRT(value, issd, imu_go, isigma_go, itau_go, imu_stop, isigma_stop, itau_stop):
    """Censored ExGaussian log-likelihood of SRRTs"""
    return stop_likelihoods.SRRT(value, issd, imu_go, isigma_go, itau_go, imu_stop, isigma_stop, itau_stop)

def cython_Inhibitions(value, imu_go, isigma_go, itau_go, imu_stop, isigma_stop, itau_stop):
    """Censored ExGaussian log-likelihood of inhibitions"""
    return stop_likelihoods.Inhibitions(value, imu_go, isigma_go, itau_go, imu_stop, isigma_stop, itau_stop)

Go_like = pm.stochastic_from_dist(name="Ex-Gauss GoRT",
                                  logp=cython_Go,
                                  dtype=np.float,
                                  mv=False)

SRRT_like = pm.stochastic_from_dist(name="CensoredEx-Gauss SRRT",
                                    logp=cython_SRRT,
                                    dtype=np.float,
                                    mv=False)

Inhibitions_like = pm.stochastic_from_dist(name="CensoredEx-Gauss Inhibittions",
                                  logp=cython_Inhibitions,
                                  dtype=np.int32,
                                  mv=False)

def Go(value, imu_go, isigma_go, itau_go):
    """Ex-Gaussian log-likelihood of GoRTs"""
    assert imu_go > 0
    assert isigma_go > 0
    assert itau_go > 0
    assert np.all(value != -999.)

    pdf = np.sum(ExGauss_pdf(value, imu_go, isigma_go, itau_go))
    if np.isinf(pdf) or np.isnan(pdf):
        return -np.inf
    else:
        return pdf

def SRRT(value, issd, imu_go, isigma_go, itau_go, imu_stop, isigma_stop, itau_stop):
    """Censored ExGaussian log-likelihood of SRRTs"""
    assert imu_go > 0
    assert isigma_go > 0
    assert itau_go > 0
    assert imu_stop > 0
    assert isigma_stop > 0
    assert itau_stop > 0
    assert np.all(value != -999.)

    pdf = np.sum(ExGauss_pdf(value, mu=imu_go, sigma=isigma_go, tau=itau_go)+ExGauss_cdf(value, mu=imu_stop+issd, sigma=isigma_stop, tau=itau_stop))
    if np.isinf(pdf) or np.isnan(pdf):
        return -np.inf
    else:
        return pdf



def Inhibitions(value, imu_go, isigma_go, itau_go, imu_stop, isigma_stop, itau_stop):
    """Censored ExGaussian log-likelihood of inhibitions"""
    assert imu_go > 0
    assert isigma_go > 0
    assert itau_go > 0
    assert imu_stop > 0
    assert isigma_stop > 0
    assert itau_stop > 0

    def CExGauss_I(value, imu_go, isigma_go, itau_go, imu_stop, isigma_stop, itau_stop, issd):
        pdf = np.exp(ExGauss_cdf(value, mu=imu_go, sigma=isigma_go, tau=itau_go))*np.exp(ExGauss_pdf(value, mu=imu_stop+issd, sigma=isigma_stop, tau=itau_stop))
        if np.isinf(pdf) or np.isnan(pdf):
            return -np.inf
        else:
            return pdf

    p = 0
    for ssd, n_trials in value:
        assert ssd != -999
        # Compute probability of single SSD
        p_ssd = np.log(quad(CExGauss_I, 0, 6000, args=(imu_go, isigma_go, itau_go, imu_stop, isigma_stop, itau_stop, ssd))[0])
        # Multiply with number of trials and add to overall p
        p += n_trials * p_ssd
    return p




class StopSignal(kabuki.Hierarchical):
    def __init__(self, data, **kwargs):
        super(StopSignal, self).__init__(data, **kwargs)


    def get_params(self):
        return [Parameter('Mu_go', lower=0, upper=np.inf),
                Parameter('Sigma_go', lower=0.01, upper=np.inf),
                Parameter('Tau_go', lower=0.01, upper=np.inf),
                Parameter('Mu_stop', lower=0, upper=np.inf),
                Parameter('Sigma_stop', lower=0.01, upper=np.inf),
                Parameter('Tau_stop', lower=0.01, upper=np.inf),
                Parameter('GoRT', is_bottom_node=True),
                Parameter('SRRT', is_bottom_node=True),
                Parameter('Inhibitions', is_bottom_node=True)]

    def get_group_node(self, param):
        if self.is_group_model:
            if param.name == 'Mu_go':
                return pm.TruncatedNormal(param.full_name, a=0, b=np.inf,
                                          mu=500, tau=1/np.sqrt(0.0001))
            if param.name == 'Sigma_go':
                return pm.TruncatedNormal(param.full_name, a=0, b=np.inf,
                                          mu=100, tau=1/np.sqrt(0.001))
            if param.name == 'Tau_go':
                return pm.TruncatedNormal(param.full_name, a=0, b=np.inf,
                                          mu=80, tau=1/np.sqrt(0.001))
            if param.name == 'Mu_stop':
                return pm.TruncatedNormal(param.full_name, a=0, b=np.inf,
                                          mu=200, tau=1/np.sqrt(0.0001))
            if param.name == 'Sigma_stop':
                return pm.TruncatedNormal(param.full_name, a=0, b=np.inf,
                                          mu=40, tau=1/np.sqrt(0.001))
            if param.name == 'Tau_stop':
                return pm.TruncatedNormal(param.full_name, a=0, b=np.inf,
                                          mu=30, tau=1/np.sqrt(0.001))
        else:
            if param.name == 'Mu_go':
                return pm.Uniform(param.full_name, lower=1, upper=1000)

            if param.name == 'Sigma_go':
                return pm.Uniform(param.full_name, lower=1, upper=300)

            if param.name == 'Tau_go':
                # CHANGED 300->2
                return pm.Uniform(param.full_name, lower=1, upper=300)

            if param.name == 'Mu_stop':
                return pm.Uniform(param.full_name, lower=1, upper=600)

            if param.name == 'Sigma_stop':
                return pm.Uniform(param.full_name, lower=1, upper=250)

            if param.name == 'Tau_stop':
                return pm.Uniform(param.full_name, lower=1, upper=250)

    def get_var_node(self, param):
        if param.name == 'Mu_go':
            return pm.Uniform(param.full_name, lower=0.01, upper=300)

        if param.name == 'Sigma_go':
            return pm.Uniform(param.full_name, lower=0.01, upper=200)

        if param.name == 'Tau_go':
            return pm.Uniform(param.full_name, lower=0.01, upper=200)

        if param.name == 'Mu_stop':
            return pm.Uniform(param.full_name, lower=0.01, upper=200)

        if param.name == 'Sigma_stop':
            return pm.Uniform(param.full_name, lower=0.01, upper=100)

        if param.name == 'Tau_stop':
            return pm.Uniform(param.full_name, lower=0.01, upper=100)

    def get_bottom_node(self, param, params):
        if param.name == 'GoRT':
            data = param.data[param.data['ss_presented'] == 0]
            return Go_like(param.full_name,
                           value=data['rt'],
                           imu_go=params['Mu_go'],
                           isigma_go=params['Sigma_go'],
                           itau_go=params['Tau_go'],
                           observed=True)

        elif param.name == 'SRRT':
            data = param.data[(param.data['ss_presented'] == 1) & (param.data['inhibited'] == 0)]

            return SRRT_like(param.full_name,
                             value=data['rt'],
                             issd=np.array(data['ssd'], dtype=np.int32),
                             imu_go=params['Mu_go'],
                             isigma_go=params['Sigma_go'],
                             itau_go=params['Tau_go'],
                             imu_stop=params['Mu_stop'],
                             isigma_stop=params['Sigma_stop'],
                             itau_stop=params['Tau_stop'],
                             observed=True)

        elif param.name == 'Inhibitions':
            data = param.data[(param.data['ss_presented'] == 1) & (param.data['inhibited'] == 1)]
            uniq_ssds = np.unique(data['ssd'])
            ssd_inhib_trials = []
            for uniq_ssd in uniq_ssds:
                ssd_inhib_trials.append((uniq_ssd, len(data[data['ssd'] == uniq_ssd])))
            ssd_inhib_trials = np.array(ssd_inhib_trials, dtype=np.int32)

            return Inhibitions_like(param.full_name,
                                    value=ssd_inhib_trials,
                                    imu_go=params['Mu_go'],
                                    isigma_go=params['Sigma_go'],
                                    itau_go=params['Tau_go'],
                                    imu_stop=params['Mu_stop'],
                                    isigma_stop=params['Sigma_stop'],
                                    itau_stop=params['Tau_stop'],
                                    observed=True)



