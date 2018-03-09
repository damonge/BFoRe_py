#!/home/ben/anaconda3/bin/python
from bfore import MapLike, SkyModel
from bfore.components import syncpl, dustmbb, cmb
from bfore.sampling import run_emcee, clean_pixels
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
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

if __name__=="__main__":
    # define true spectral parameters
    beta_s_true = -3.
    beta_d_true = 1.6
    T_d_true = 20.
    nu_ref_s = 23.
    nu_ref_d = 353.
    nside_spec = 2
    nside = 8
    true_params = {
        'beta_d': beta_d_true,
        'T_d': T_d_true,
        'beta_s': beta_s_true,
        'nu_ref_s': nu_ref_s,
        'nu_ref_d': nu_ref_d
    }

    components = ["syncpl", "dustmbb", "cmb"]

    nus = [10., 20., 25., 45., 90., 100., 143., 217., 300., 350., 400., 500.]
    sigmas = [1 * sig for sig in [110., 50., 36., 8., 4, 4, 10.1, 20., 25., 30., 40., 50.]]

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

    temp_s = np.ones_like(temp_s) * 15.
    temp_d = np.ones_like(temp_s) * 20.
    temp_c = np.ones_like(temp_s) * 25.

    # the synchrotron and dust signals separates
    synch = np.array([temp_s * syncpl(np.array([nu]), beta_s=beta_s_true, nu_ref_s=nu_ref_s) for nu in nus])
    dust = np.array([temp_d * dustmbb(np.array([nu]), beta_d=beta_d_true, T_d=T_d_true, nu_ref_d=nu_ref_d) for nu in nus])
    cmbs = np.array([temp_c * cmb(np.array([nu])) for nu in nus])

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

    # likelihood setup.
    config_dict = {
        "nus": nus,
        "fpaths_mean": fpaths_mean,
        "fpaths_vars": fpaths_vars,
        "nside_spec": nside_spec,
        "fixed_pars": {"nu_ref_d": 353., "nu_ref_s": 23., "T_d": 20.},
        "var_pars": ["beta_s", "beta_d"]
            }
    skymodel = SkyModel(components)
    ml = MapLike(config_dict, skymodel)

    # sampler setup
    sampler_args = {
        "ndim": 2,
        "nwalkers": 10,
        "nsamps": 300,
        "nburn": 50,
        "pos0": [-3., 1.6]
    }

    gen = ml.split_data()
    for (mean, var) in gen:
        print(ml.chi2((-3, 1.6), mean, var))

    # do the cleaning over a list of pixels
    ipixs = [10, 11, 12, 13]
    samples_list = clean_pixels(ml, run_emcee, ipix=ipixs, **sampler_args)

    # plot the results.
    for ipix, samples in zip(ipixs, samples_list):
        beta_s_mcmc, beta_d_mcmc= map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84], axis=0)))
        print(beta_s_mcmc, beta_d_mcmc)
        fig = corner.corner(samples, labels=[r"$\beta_s$", r"$\beta_d$"],
                          truths=[beta_s_true, beta_d_true])
        fig.savefig("fit_ipix{:04d}.pdf".format(ipix))
        plt.show()
