#!/home/ben/anaconda3/bin/python
from bfore import MapLike, SkyModel
from bfore.components import syncpl, dustmbb
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
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

def test_maplike_grid():

    beta_s_true = -3.
    beta_d_true = 1.6
    T_d_true = 20.
    nu_ref_s = 23.
    nu_ref_d = 353.
    nside_spec = 2
    nside = 8

    components = ["syncpl", "dustmbb"]

    nus = [10., 20., 25., 45., 90., 100., 143., 217., 300., 350., 400., 500.]
    sigmas = [1. * sig for sig in [110., 50., 36., 8., 4, 4, 10.1, 20., 25., 30., 40., 50.]]

    synch = [15 * np.ones((3, hp.nside2npix(nside))) * syncpl(nu, beta_s=beta_s_true, nu_ref_s=nu_ref_s) for nu in nus]
    dust = [15. * np.ones((3, hp.nside2npix(nside))) * dustmbb(nu, beta_d=beta_d_true, T_d=T_d_true, nu_ref_d=nu_ref_d) for nu in nus]
    noise = [add_noise(sig, nside) for sig in sigmas]
    maps = [s + d + n  for d, s, n in zip(dust, synch, noise)]
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
        "var_pars": ["beta_d", "T_d", "beta_s"]
            }

    # initialize sky model and likelihood
    skymodel = SkyModel(components)
    ml = MapLike(config_dict, skymodel)
    gen = ml.split_data()

    # compute likelihood on grid of parameters
    params_dicts = []
    nsamp = 32
    beta_d = np.linspace(-0.1, 0.1, nsamp) + beta_d_true
    T_d = np.linspace(-2, 2, nsamp) + T_d_true
    beta_s = np.linspace(-0.1, 0.1, nsamp) + beta_s_true
    lkl = np.zeros((nsamp, nsamp, nsamp))

    # cycle through data one big pixel at a time
    for (mean, var) in gen:
        for i, b_d in enumerate(beta_d):
            for j, T in enumerate(T_d):
                for k, b_s in enumerate(beta_s):
                    params = {
                        'beta_d': b_d,
                        'T_d': T,
                        'beta_s': b_s,
                        'nu_ref_s': nu_ref_s,
                        'nu_ref_d': nu_ref_d
                    }
                    lkl[i, j, k] = ml.marginal_spectral_likelihood(mean, var, params)

        # plot 2d posteriors
        plt.imshow(np.sum(lkl, axis=0), origin='lower', aspect='auto', extent=(T_d.min(), T_d.max(), beta_s.min(), beta_s.max()))
        plt.title("T_d - beta_s")
        plt.xlabel(r"T_d")
        plt.ylabel(r"beta_s")
        plt.colorbar(label=r"$F^T N_T^{-1} F$")
        plt.show()

        plt.imshow(np.sum(lkl, axis=1), origin='lower', aspect='auto', extent=(beta_d.min(), beta_d.max(), beta_s.min(), beta_s.max()))
        plt.title("beta_d - beta_s")
        plt.xlabel(r"beta_d")
        plt.ylabel(r"beta_s")
        plt.colorbar(label=r"$F^T N_T^{-1} F$")
        plt.show()

        plt.imshow(np.sum(lkl, axis=2), origin='lower', aspect='auto', extent=(beta_d.min(), beta_d.max(), T_d.min(), T_d.max()))
        plt.title("beta_d - T_d")
        plt.xlabel(r"beta_d")
        plt.ylabel(r"T_d")
        plt.colorbar(label=r"$F^T N_T^{-1} F$")
        plt.show()

        # plot 1d posteriors
        beta_s_1d = np.sum(lkl, axis=(0, 1))
        T_d_1d = np.sum(lkl, axis=(0, 2))
        beta_d_1d = np.sum(lkl, axis=(1, 2))

        plt.plot(beta_s, beta_s_1d)
        plt.axvline(beta_s_true, color='k', linestyle='--')
        plt.title("beta_s, max={:f}".format(beta_s[np.argmax(beta_s_1d)]))
        plt.show()

        plt.plot(T_d, T_d_1d)
        plt.axvline(T_d_true, color='k', linestyle='--')
        plt.title("T_d, max={:f}".format(T_d[np.argmax(T_d_1d)]))
        plt.show()

        plt.plot(beta_d, beta_d_1d)
        plt.axvline(beta_d_true, color='k', linestyle='--')
        plt.title("beta_d, max={:f}".format(beta_d[np.argmax(beta_d_1d)]))
        plt.show()
    return

if __name__=="__main__":
    test_maplike_grid()
