from __future__ import absolute_import
from unittest import TestCase
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from .setup_maplike import setup_maplike
from bfore.sampling import clean_pixels, run_emcee, run_minimize
import corner

class test_MapLike(TestCase):
    def setUp(self):
        self.maplike, self.true_params = setup_maplike()
        return

    def test_chi2(self):
        print("For true parameters")
        print("pval \t chi2 per dof")
        pval = self.maplike.pval(self.true_params)
        chi2perdof= self.maplike.chi2perdof(np.array(self.true_params)+0.0)
        print("{:01.2E} \t {:01.2E}".format(pval, chi2perdof))
        return

    def test_minimize(self):
        print('Finding maximum likelihood')
        sampler_args = {
            "method":'Powell',
            "tol":None,
            "callback":None,
            "options":{'xtol':1E-4,'ftol':1E-4,'maxiter':None,'maxfev':None,'direc':None}
            }
        samples=clean_pixels(self.maplike,run_minimize,**sampler_args)

        print("Results: ",)
        if samples[1] :
            print(" Successful maximization")
        else :
            print(" Unsuccessful maximization")
        print(" Param truth: ",self.true_params)
        print(" Param ML: ",samples[0])

    def test_emcee(self):
        # Calculate the p value and reduced chi squred for the true parameter values
        # in the 4 pixels above.

        sampler_args = {
            "nwalkers": 20,
            "nsamps": 500,
            "nburn": 100,
            "verbose": True
        }
        samples=clean_pixels(self.maplike,run_emcee,**sampler_args)
        ranges = lambda v: (v[1], v[2]-v[1], v[1]-v[0])
        params_mcmc = list(map(ranges, zip(*np.percentile(samples, [16, 50, 84], axis=0))))
        print("Results: \n", )
        print("Param medians: ", [p[0] for p in params_mcmc])
        print("Param spread, 14th to 84th percentile: ", [p[1] + p[2] for p in params_mcmc])
        labels = [r"$\beta_s$", r"$\beta_d$", r"$T_d$", r"$\beta_c$"]
        fig = corner.corner(samples, labels=labels, truths=self.true_params)
        plt.show()
