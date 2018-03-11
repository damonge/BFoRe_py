#!/home/ben/anaconda3/bin/python
from bfore.sampling import run_emcee, clean_pixels
from tests.setup_maplike import setup_maplike
import numpy as np
import matplotlib.pyplot as plt
import corner

if __name__=="__main__":
    ipixs = [10, 11, 12, 13]
    ml, true_params = setup_maplike()
    # Calculate the p value and reduced chi squred for the true parameter values
    # in the 4 pixels above.
    pixel_data = ml.split_data(ipix=ipixs)
    for (mean, var) in pixel_data:
        print("p-value for true params: ", ml.pval(true_params, mean, var))
        print("chi2 per dof for true params: ", ml.chi2perdof(true_params, mean, var))

    # do the cleaning over the same list of pixels.
    sampler_args = {
        "ndim": len(true_params),
        "nwalkers": 10,
        "nsamps": 300,
        "nburn": 50,
        "pos0": true_params
    }
    for samples in clean_pixels(ml, run_emcee, ipix=ipixs, **sampler_args):
        # Get the median and percentile ranges from the posterior samples
        ranges = lambda v: (v[1], v[2]-v[1], v[1]-v[0])
        params_mcmc = list(map(ranges, zip(*np.percentile(samples, [16, 50, 84], axis=0))))
        print("Results: \n", )
        print("Param medians: ", [p[0] for p in params_mcmc])
        print("Param spread, 14th to 84th percentile: ", [p[1] + p[2] for p in params_mcmc])
        labels = [r"$\beta_s$", r"$\beta_d$", r"$T_d$"]
        fig = corner.corner(samples, labels=labels, truths=true_params)
        plt.show()
