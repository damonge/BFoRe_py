#!/home/ben/anaconda3/bin/python
from bfore import MapLike, SkyModel
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from os.path import join, abspath

def pixel_var(sigma_amin, nside):
    npix = hp.nside2npix(nside)
    amin_sq_per_pix = 4 * np.pi * (180. * 60. / np.pi) ** 2 / npix
    pixel_var = sigma_amin ** 2 / amin_sq_per_pix
    return pixel_var

def test_setup(params):
    nside = 8
    nus = [10., 30., 45., 90., 100., 143., 350., 450.]
    ells = np.linspace(0, 3 * nside, 3 * nside + 1)
    cltt_ref = np.zeros_like(ells)
    cltt_ref[2:] = 5. * (ells[2:] / 80.) ** -3.2
    cltt = lambda nu: cltt_ref * (nu / 23.) ** -3.
    maps = [hp.synfast([cltt(nu), 0.1 * cltt(nu), 0.2 * cltt(nu), 0.3 * cltt(nu)], nside, verbose=False) for nu in nus]
    vars = [1. / pixel_var(4., nside) * np.ones_like(m) for m in maps]
    for m, v in zip(maps, vars):
        print(m[0, 0], v[0, 0])
    exit()
    test_dir = abspath("test_data")
    fpaths_mean = [join(test_dir, "mean_nu{:03d}.fits".format(int(nu))) for nu in nus]
    fpaths_vars = [join(test_dir, "vars_nu{:03d}.fits".format(int(nu))) for nu in nus]

    for nu, m, fm, v, fv in zip(nus, maps, fpaths_mean, vars, fpaths_vars):
        hp.write_map(fm, m, overwrite=True)
        hp.write_map(fv, v, overwrite=True)

    config_dict = {
        "nus": nus,
        "fpaths_mean": fpaths_mean,
        "fpaths_vars": fpaths_vars,
        "nside_spec": 2,
        "initial_param_guess": params,
        "var_pars": ["beta_d", "T_d", "beta_s"]
            }
    return config_dict

def test_maplike(components, params):
    skymodel = SkyModel(components)
    ml = MapLike(config_dict, skymodel)
    print(ml.nus)
    print(ml.fpaths_mean)
    print(ml.fpaths_vars)
    print(ml.data_mean.shape)
    print(ml.data_vars.shape)
    print(ml.f_matrix(params))
    print(ml.f_matrix(params))
    gen = ml.split_data()
    for (mean, var) in gen:
        print(mean.shape)
        covar = ml.get_amplitude_covariance(var, params)
        print("amp covar shape: ", covar.shape)
        print("amp covar: ", covar[0, 0 , :, :])
        print("Pixel var shape: ", var.shape)
        amp_mean = ml.get_amplitude_mean(mean, var, params, None)
        print("amp mean shape: ", amp_mean.shape)
        lkl = ml.marginal_spectral_likelihood(mean, var, params)
        print("likelihood: ", lkl)
        chain = ml.sample_marginal_spectral_likelihood(mean, var, 100)
    return

def test_skymodel(components, params):
    # initialize sky model with a list of component sed names.
    skymodel = SkyModel(components)
    # calculate the F matrix over a set of frequencies.
    freqs = np.logspace(1, 2.5, 100)
    fnus = skymodel.fnu(freqs, params)
    # print some information about the SEDs we are using
    skymodel.get_model_parameters()
    # plot the seds we just calculated.
    fig, ax = plt.subplots(1, 1)
    ax.loglog(freqs, fnus[0, 0], label='cmb')
    ax.loglog(freqs, fnus[0, 1], label='sync')
    #ax.loglog(freqs, fnus[0, 2], label='dust')
    ax.legend()
    return

if __name__=="__main__":
    components = ["syncpl"]
    params = {
    "beta_s": -3.1,
    "beta_d": 1.5,
    "T_d": 19,
    "nu_ref_s": 23.,
    "nu_ref_d": 353.,
    }
    test_skymodel(components, params)

    config_dict = test_setup(params)
    test_maplike(components, params)
