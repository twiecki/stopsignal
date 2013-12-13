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
from collections import OrderedDict

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
        self.group_only_nodes = kwargs.pop('group_only_nodes', ())
        super(StopSignal, self).__init__(data, **kwargs)

    def create_ss_knode(self, knodes):
        ss_parents = OrderedDict()
        ss_parents['imu_go'] = knodes['mu_go_bottom']
        ss_parents['isigma_go'] = knodes['sigma_go_bottom']
        ss_parents['itau_go'] = knodes['tau_go_bottom']
        ss_parents['imu_stop'] = knodes['mu_stop_bottom']
        ss_parents['isigma_stop'] = knodes['sigma_stop_bottom']
        ss_parents['itau_stop'] = knodes['tau_stop_bottom']

        go_like = KnodeGo(Go_like, 'go_like', col_name='rt', observed=True, imu_go=ss_parents['imu_go'], isigma_go=ss_parents['isigma_go'], itau_go=ss_parents['itau_go'])
        srrt_like = KnodeSRRT(SRRT_like, 'srrt_like', col_name='rt', observed=True, **ss_parents)
        inhibitions_like = KnodeInhibitions(Inhibitions_like, 'inhibitions_like', col_name='rt', observed=True, **ss_parents)

        return [go_like, srrt_like, inhibitions_like]

    def create_knodes(self):
        knodes = OrderedDict()
        knodes.update(self._create_family_trunc_normal('mu_go', lower=1e-3, upper=1e3, value=400, std_lower=0.01, std_upper=300, std_value=50))
        knodes.update(self._create_family_trunc_normal('sigma_go', lower=1, upper=500, value=50, std_lower=0.01, std_upper=200, std_value=20))
        knodes.update(self._create_family_trunc_normal('tau_go', lower=1, upper=500, value=50, std_lower=0.01, std_upper=200, std_value=20))
        knodes.update(self._create_family_trunc_normal('mu_stop', lower=1e-3, upper=600, value=200, std_lower=0.01, std_upper=300, std_value=40))
        knodes.update(self._create_family_trunc_normal('sigma_stop', lower=1, upper=350, value=30, std_lower=0.01, std_upper=200, std_value=20))
        knodes.update(self._create_family_trunc_normal('tau_stop', lower=1, upper=350, value=30, std_lower=0.01, std_upper=200, std_value=20))

        likelihoods = self.create_ss_knode(knodes)

        return knodes.values() + likelihoods

    def _create_family_trunc_normal(self, name, value=0, lower=None,
                                   upper=None, std_lower=1e-10,
                                   std_upper=100, std_value=.1):
        """Similar to _create_family_normal() but creates a Uniform
        group distribution and a truncated subject distribution.

        See _create_family_normal() help for more information.

        """
        knodes = OrderedDict()

        if self.is_group_model and name not in self.group_only_nodes:
            g = Knode(pm.Uniform, '%s' % name, lower=lower,
                      upper=upper, value=value, depends=self.depends[name])

            depends_std = self.depends[name] if self.std_depends else ()
            std = Knode(pm.Uniform, '%s_std' % name, lower=std_lower,
                        upper=std_upper, value=std_value, depends=depends_std)
            tau = Knode(pm.Deterministic, '%s_tau' % name,
                        doc='%s_tau' % name, eval=lambda x: x**-2, x=std,
                        plot=False, trace=False, hidden=True)
            subj = Knode(pm.TruncatedNormal, '%s_subj' % name, mu=g,
                         tau=tau, a=lower, b=upper, value=value,
                         depends=('subj_idx',), subj=True, plot=self.plot_subjs)

            knodes['%s'%name] = g
            knodes['%s_std'%name] = std
            knodes['%s_tau'%name] = tau
            knodes['%s_bottom'%name] = subj

        else:
            subj = Knode(pm.Uniform, name, lower=lower,
                         upper=upper, value=value,
                         depends=self.depends[name])
            knodes['%s_bottom'%name] = subj

        return knodes
