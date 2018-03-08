#!/home/ben/anaconda3/bin/python
from bfore import MapLike, SkyModel
from bfore.components import syncpl, dustmbb, cmb
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import emcee
from schwimmbad import MultiPool
from functools import partial
import corner
from os.path import join, abspath

def pixel_var(sigma_amin, nside):
    npix = hp.nside2npix(nside)
    amin_sq_per_pix = 4 * np.pi * (180. * 60. / np.pi) ** 2 / npix
    pixel_var = sigma_amin ** 2 / amin_sq_per_pix
    return pixel_var

def add_noise(sigma_amin, nside):
    sigma_pix = np.sqrt(pixel_var(sigma_amin, nside))
    noise = np.random.randn(3, hp.nside2npix(nside)) * sigma_pix
    return noise

def run_sampler(data, pos, func, nwalkers=10, ndim=3, nsamps=100, nburn=50):
    print("Sampling")
    (mean, var) = data
    sampler = emcee.EnsembleSampler(nwalkers, ndim, func, args=(mean, var))
    sampler.run_mcmc(pos, nsamps)
    samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
    print("Finished sampling.")
    return samples


if __name__=="__main__":
    # define true spectral parameters
    beta_s_true = -3.
    beta_d_true = 1.6
    T_d_true = 20.
    nu_ref_s = 23.
    nu_ref_d = 353.
    nside_spec = 16
    nside = 256
    true_params = {
        'beta_d': beta_d_true,
        'T_d': T_d_true,
        'beta_s': beta_s_true,
        'nu_ref_s': nu_ref_s,
        'nu_ref_d': nu_ref_d
    }

    components = ["syncpl", "dustmbb", "cmb"]

    nus = [10., 20., 25., 45., 90., 100., 143., 217., 300., 350., 400., 500.]
    sigmas = [1. * sig for sig in [110., 50., 36., 8., 4, 4, 10.1, 20., 25., 30., 40., 50.]]

    # generate fake synch and dust as GRFs
    ells = np.linspace(0, 3 * nside, 3 * nside + 1)
    cl_s = np.zeros_like(ells)
    cl_d = np.zeros_like(ells)
    cl_s[2:] = 10. * (ells[2:] / 80.) ** - 3.2
    cl_d[2:] = 10. * (ells[2:] / 80.) ** - 3.2

    # the templates of dust and synchrotron at their reference frequencies
    temp_s = np.array(hp.synfast([cl_s, cl_s, cl_s, cl_s], nside, verbose=False, pol=True))
    temp_d = np.array(hp.synfast([cl_d, cl_d, cl_d, cl_d], nside, verbose=False, pol=True))
    temp_c = np.array(hp.synfast([cl_d, cl_d, cl_d, cl_d], nside, verbose=False, pol=True))

    # the synchrotron and dust signals separates
    synch = np.array([temp_s * syncpl(nu, beta_s=beta_s_true, nu_ref_s=nu_ref_s) for nu in nus])
    dust = np.array([temp_d * dustmbb(nu, beta_d=beta_d_true, T_d=T_d_true, nu_ref_d=nu_ref_d) for nu in nus])
    cmbs = np.array([temp_c * cmb(nu) for nu in nus])

    # the noise maps
    noise = [add_noise(sig, nside) for sig in sigmas]

    # these are the simulated observations mean and variance
    # synch + dust + noise
    maps = [d + s + c + n  for d, s, c, n in zip(dust, synch, cmbs, noise)]
    # inverse pixel noise variance
    vars = [np.ones((3, hp.nside2npix(nside))) / pixel_var(sig, nside) for sig in sigmas]

    # Save maps
    test_dir = abspath("test_data")
    fpaths_mean = [join(test_dir, "mean_nu{:03d}.fits".format(int(nu))) for nu in nus]
    fpaths_vars = [join(test_dir, "vars_nu{:03d}.fits".format(int(nu))) for nu in nus]
    for nu, m, fm, v, fv in zip(nus, maps, fpaths_mean, vars, fpaths_vars):
        hp.write_map(fm, m, overwrite=True)
        hp.write_map(fv, v, overwrite=True)

    # start likelihood setup.
    config_dict = {
        "nus": nus,
        "fpaths_mean": fpaths_mean,
        "fpaths_vars": fpaths_vars,
        "nside_spec": nside_spec,
            }

    # initialize sky model and likelihood
    skymodel = SkyModel(components)
    ml = MapLike(config_dict, skymodel)
    gen = ml.split_data(ipix=[10, 11])

    sampler_args = {
        "ndim": 3,
        "nwalkers": 100,
        "nsamps": 500,
        "nburn": 50,
    }

    pos = [[beta_s_true, beta_d_true, T_d_true] + 1e-2 * np.random.randn(sampler_args['ndim']) for i in range(sampler_args['nwalkers'])]
    sample_func = partial(run_sampler, pos=pos, func=ml.marginal_spectral_likelihood, **sampler_args)
    with MultiPool() as pool:
        samples_list = list(pool.map(sample_func, gen))

    for samples in samples_list:
        beta_s_mcmc, beta_d_mcmc, T_d_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84], axis=0)))
        print(beta_s_mcmc, beta_d_mcmc, T_d_mcmc)
        print(samples.shape)
        fig = corner.corner(samples, labels=[r"$\beta_s$", r"$\beta_d$", r"$T_d$"],
                          truths=[beta_s_true, beta_d_true, T_d_true])
        plt.show()
