from __future__ import absolute_import
from unittest import TestCase
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from .setup_maplike import setup_maplike
from bfore.sampling import clean_pixels, run_emcee
import corner

class test_MapLike(TestCase):
    def setUp(self):
        self.maplike, self.true_params = setup_maplike()
        self.data = list(self.maplike.split_data())
        return

    def test_chi2(self):
        pixel_data = self.maplike.split_data()
        print("For true parameters")
        print("ipix \t pval \t chi2 per dof")
        for ip, (mean, var) in enumerate(pixel_data):
            pval = self.maplike.pval(self.true_params, mean, var)
            chi2perdof= self.maplike.chi2perdof(self.true_params, mean, var)
            print("{:04d} \t {:01.2f} \t {:01.2f}".format(ip, pval, chi2perdof))
        return


    def test_sampling(self):
        ipixs = [10, 11, 12, 13]
        # Calculate the p value and reduced chi squred for the true parameter values
        # in the 4 pixels above.

        sampler_args = {
            "ndim": len(self.true_params),
            "nwalkers": 20,
            "nsamps": 500,
            "nburn": 100,
            "pos0": self.true_params
        }
        for samples in clean_pixels(self.maplike, run_emcee, ipix=ipixs, **sampler_args):
            # Get the median and percentile ranges from the posterior samples
            ranges = lambda v: (v[1], v[2]-v[1], v[1]-v[0])
            params_mcmc = list(map(ranges, zip(*np.percentile(samples, [16, 50, 84], axis=0))))
            print("Results: \n", )
            print("Param medians: ", [p[0] for p in params_mcmc])
            print("Param spread, 14th to 84th percentile: ", [p[1] + p[2] for p in params_mcmc])
            labels = [r"$\beta_s$", r"$\beta_d$", r"$T_d$", r"$\beta_c$"]
            fig = corner.corner(samples, labels=labels, truths=self.true_params)
            plt.show()
        return
