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

from kabuki import Hierarchical, Knode

import stop_likelihoods

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
class KnodeGo(Knode):
    def create_node(self, node_name, kwargs, data):
        new_data = data[data['ss_presented'] == 0]
        kwargs['value'] = new_data['rt']
        return self.pymc_node(name=node_name, **kwargs)

class KnodeSRRT(Knode):
    def create_node(self, node_name, kwargs, data):
        new_data = data[(data['ss_presented'] == 1) & (data['inhibited'] == 0)]
        kwargs['value'] = new_data['rt']
        kwargs['issd'] = np.array(new_data['ssd'], dtype=np.int32)
        return self.pymc_node(name=node_name, **kwargs)

class KnodeInhibitions(Knode):
    def create_node(self, node_name, kwargs, data):
        new_data = data[(data['ss_presented'] == 1) & (data['inhibited'] == 1)]
        uniq_ssds = np.unique(new_data['ssd'])
        ssd_inhib_trials = []
        for uniq_ssd in uniq_ssds:
            ssd_inhib_trials.append((uniq_ssd, len(new_data[new_data['ssd'] == uniq_ssd])))
        ssd_inhib_trials = np.array(ssd_inhib_trials, dtype=np.int32)
        kwargs['value'] = ssd_inhib_trials
        return self.pymc_node(name=node_name, **kwargs)

class StopSignal(Hierarchical):
    def __init__(self, data, **kwargs):
        super(StopSignal, self).__init__(data, **kwargs)

    def create_knodes(self):

        if self.is_group_model:
            mu_go_mean = Knode(pm.TruncatedNormal, 'mu_go_mean', a=0, b=np.inf, mu=500, tau=0.0001)
            mu_go_sd = Knode(pm.Uniform, 'mu_go_sd', lower=0.01, upper=300)
            mu_go_tau = Knode(pm.Deterministic, 'mu_go_tau', eval=lambda x: x**-2, x=mu_go_sd)
            mu_go_subj = Knode(pm.TruncatedNormal, 'mu_go_subj', a=0, b=np.inf, mu=mu_go_mean, tau=mu_go_tau, depends=('subj_idx',), subj=True)

            sigma_go_mean = Knode(pm.TruncatedNormal, 'sigma_go_mean', a=0, b=np.inf, mu=100, tau=0.001)
            sigma_go_sd = Knode(pm.Uniform, 'sigma_go_sd', lower=0.01, upper=200)
            sigma_go_tau = Knode(pm.Deterministic, 'sigma_go_tau', eval=lambda x: x**-2, x=sigma_go_sd)
            sigma_go_subj = Knode(pm.TruncatedNormal, 'sigma_go_subj', a=1, b=np.inf, mu=sigma_go_mean, tau=sigma_go_tau, depends=('subj_idx',), subj=True)

            tau_go_mean = Knode(pm.TruncatedNormal, 'tau_go_mean', a=0, b=np.inf, mu=80, tau=0.001)
            tau_go_sd = Knode(pm.Uniform, 'tau_go_sd', lower=0.01, upper=200)
            tau_go_tau = Knode(pm.Deterministic, 'tau_go_tau', eval=lambda x: x**-2, x=tau_go_sd)
            tau_go_subj = Knode(pm.TruncatedNormal, 'tau_go_subj', a=1, b=np.inf, mu=tau_go_mean, tau=tau_go_tau, depends=('subj_idx',), subj=True)

            mu_stop_mean = Knode(pm.TruncatedNormal, 'mu_stop_mean', a=0, b=np.inf, mu=200, tau=0.0001)
            mu_stop_sd = Knode(pm.Uniform, 'mu_stop_sd', lower=0.01, upper=200)
            mu_stop_tau = Knode(pm.Deterministic, 'mu_stop_tau', eval=lambda x: x**-2, x=mu_stop_sd)
            mu_stop_subj = Knode(pm.TruncatedNormal, 'mu_stop_subj', a=0, b=np.inf, mu=mu_stop_mean, tau=mu_stop_tau, depends=('subj_idx',), subj=True)

            sigma_stop_mean = Knode(pm.TruncatedNormal, 'sigma_stop_mean', a=0, b=np.inf, mu=40, tau=0.001)
            sigma_stop_sd = Knode(pm.Uniform, 'sigma_stop_sd', lower=0.01, upper=100)
            sigma_stop_tau = Knode(pm.Deterministic, 'sigma_stop_tau', eval=lambda x: x**-2, x=sigma_stop_sd)
            sigma_stop_subj = Knode(pm.TruncatedNormal, 'sigma_stop_subj', a=1, b=np.inf, mu=sigma_stop_mean, tau=sigma_stop_tau, depends=('subj_idx',), subj=True)

            tau_stop_mean = Knode(pm.TruncatedNormal, 'tau_stop_mean', a=0, b=np.inf, mu=30, tau=0.001)
            tau_stop_sd = Knode(pm.Uniform, 'tau_stop_sd', lower=0.01, upper=100)
            tau_stop_tau = Knode(pm.Deterministic, 'tau_stop_tau', eval=lambda x: x**-2, x=tau_stop_sd)
            tau_stop_subj = Knode(pm.TruncatedNormal, 'tau_stop_subj', a=1, b=np.inf, mu=tau_stop_mean, tau=tau_stop_tau, depends=('subj_idx',), subj=True)

            ##Error if I don't supply an existing col_name. However, it doesn't seem to use rt col as data, rather what I have specified above.
            go_like = KnodeGo(Go_like, 'go_like', imu_go = mu_go_subj, isigma_go = sigma_go_subj, itau_go = tau_go_subj, col_name='rt', observed=True)
            srrt_like = KnodeSRRT(SRRT_like, 'srrt_like', imu_go = mu_go_subj, isigma_go = sigma_go_subj, itau_go = tau_go_subj, imu_stop = mu_stop_subj,isigma_stop = sigma_stop_subj, itau_stop = tau_stop_subj, col_name='rt', observed=True)
            inhibitions_like = KnodeInhibitions(Inhibitions_like, 'inhibitions_like', imu_go = mu_go_subj, isigma_go = sigma_go_subj, itau_go = tau_go_subj,imu_stop = mu_stop_subj, isigma_stop = sigma_stop_subj, itau_stop = tau_stop_subj, col_name='rt', observed=True)
           
            return [mu_go_mean, mu_go_sd, mu_go_tau, mu_go_subj, sigma_go_mean, sigma_go_sd, sigma_go_tau, sigma_go_subj, tau_go_mean, tau_go_sd, tau_go_tau, tau_go_subj,
                    mu_stop_mean, mu_stop_sd, mu_stop_tau, mu_stop_subj, sigma_stop_mean, sigma_stop_sd, sigma_stop_tau, sigma_stop_subj, tau_stop_mean, tau_stop_sd, tau_stop_tau, tau_stop_subj,
                    go_like,srrt_like,inhibitions_like]
        else:
            mu_go_subj = Knode(pm.Uniform, 'mu_go_subj', lower=0, upper=1000)
            sigma_go_subj = Knode(pm.Uniform, 'sigma_go_subj', lower=1, upper=300)
            tau_go_subj = Knode(pm.Uniform, 'tau_go_subj', lower=1, upper=300)

            mu_stop_subj = Knode(pm.Uniform, 'mu_stop_subj', lower=0,upper=600)
            sigma_stop_subj = Knode(pm.Uniform, 'sigma_stop_subj', lower=1,upper=250)
            tau_stop_subj = Knode(pm.Uniform, 'tau_stop_subj', lower=1,upper=250)

            go_like = KnodeGo(Go_like, 'go_like', imu_go = mu_go_subj, isigma_go = sigma_go_subj, itau_go = tau_go_subj, col_name='rt', observed=True)
            srrt_like = KnodeSRRT(SRRT_like, 'srrt_like', imu_go = mu_go_subj, isigma_go = sigma_go_subj, itau_go = tau_go_subj, imu_stop = mu_stop_subj, isigma_stop = sigma_stop_subj, itau_stop = tau_stop_subj, col_name='rt', observed=True)
            inhibitions_like = KnodeInhibitions(Inhibitions_like, 'inhibitions_like', imu_go = mu_go_subj, isigma_go = sigma_go_subj, itau_go = tau_go_subj, imu_stop = mu_stop_subj, isigma_stop = sigma_stop_subj, itau_stop = tau_stop_subj, col_name='rt', observed=True)
           
            return [mu_go_subj, sigma_go_subj, tau_go_subj, mu_stop_subj, sigma_stop_subj, tau_stop_subj, go_like, srrt_like, inhibitions_like]
