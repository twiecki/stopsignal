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
        super(StopSignal, self).__init__(data, **kwargs)

    def create_ss_knode(self, knodes):
        ss_parents = OrderedDict()
        # TW: The keys in the ss_parents dict have to the names of the arguments you use in the likelihoods (prefixed with an 'i')
        ss_parents['imu_go'] = knodes['mu_go_bottom']
        ss_parents['isigma_go'] = knodes['sigma_go_bottom']
        ss_parents['itau_go'] = knodes['tau_go_bottom']
        ss_parents['imu_stop'] = knodes['mu_stop_bottom']
        ss_parents['isigma_stop'] = knodes['sigma_stop_bottom']
        ss_parents['itau_stop'] = knodes['tau_stop_bottom']

        #go_like = KnodeGo(Go_like, 'go_like', imu_go = 'mu_go', isigma_go = 'sigma_go', itau_go = 'tau_go', col_name='rt', observed=True)
        #srrt_like = KnodeSRRT(SRRT_like, 'srrt_like', imu_go = 'mu_go', isigma_go = 'sigma_go', itau_go = 'tau_go', imu_stop = 'mu_stop', isigma_stop = 'sigma_stop',itau_stop = 'tau_stop', col_name='rt', observed=True)
        #inhibitions_like = KnodeInhibitions(Inhibitions_like, 'inhibitions_like',  imu_go = 'mu_go', isigma_go = 'sigma_go', itau_go = 'tau_go', imu_stop = 'mu_stop', isigma_stop = 'sigma_stop', itau_stop = 'tau_stop', col_name='rt', observed=True)
        # TW: This was a bug, you passed strings before but you need to pass the knode (which is an element in the ss_parents dictionary).
        go_like = KnodeGo(Go_like, 'go_like', col_name='rt', observed=True, imu_go=ss_parents['imu_go'], isigma_go=ss_parents['isigma_go'], itau_go=ss_parents['itau_go'])
        srrt_like = KnodeSRRT(SRRT_like, 'srrt_like', col_name='rt', observed=True, **ss_parents)
        inhibitions_like = KnodeInhibitions(Inhibitions_like, 'inhibitions_like', col_name='rt', observed=True, **ss_parents)

        return [go_like,srrt_like,inhibitions_like]

    def create_knodes(self):
        knodes = OrderedDict()
        knodes.update(self.create_family_trunc_normal('mu_go', lower=1e-3, upper=1e3, value=500,var_lower=0.01, var_upper=300, var_value=50))
        knodes.update(self.create_family_trunc_normal('sigma_go', lower=1, upper=300, value=50,var_lower=0.01, var_upper=200, var_value=20))
        knodes.update(self.create_family_trunc_normal('tau_go', lower=1, upper=300, value=40,var_lower=0.01, var_upper=200, var_value=20))
        knodes.update(self.create_family_trunc_normal('mu_stop', lower=1e-3, upper=600, value=200,var_lower=0.01, var_upper=200, var_value=40))
        knodes.update(self.create_family_trunc_normal('sigma_stop', lower=1, upper=250, value=30,var_lower=0.01, var_upper=100, var_value=20))
        knodes.update(self.create_family_trunc_normal('tau_stop', lower=1, upper=250, value=30,var_lower=0.01, var_upper=100, var_value=20))
        #what is value/ var_value?-->start values?
        # TW: yes. starting value for mean and var_value is starting value for variance nodes.
        #Can I generate uniform random numbers instead of giving it a fixed value? How to set start values for the individual subject parameters?
        # TW: you should be able to use np.random.rand() and supply that to as value
        #In WinBUGS implementation, group-level means (g) are (truncated) normally distributed; here they are Uniform (in hierarchical.py).
        #Is it possible to overwrite that?
        # TW: You would need to create your own create_family convenience function (see the code, it's basically doing the same thing as you are doing in stopsignal.py by hand).
        # TW: I would consider reformulating the model however because truncated normal distributions are not as easy to sample from and you can't use gibbs sampling. What we did in the new HDDM version was to use Normal group distributions and Normal subject distributions and then transform them appropriately (e.g. exp if values are positive; for this you can use the convenience function create_family_exp(). See hddm.models.hddm_transformed.py for an example). This made convergence so much better and faster! Especially because convergence in the examples I ran is not great.

        likelihoods = self.create_ss_knode(knodes)#What does this do exactly? Can I pass on the likleihoods like that?
        # TW: We just need to return the knodes. I save them in a dictionary above so that I can fetch them out in create_ss_knode(), but Hierarchical just requires a list so we can just save your likelihood knodes and return them together with the stochastic knodes.

        return knodes.values() + likelihoods

