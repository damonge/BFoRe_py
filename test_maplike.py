#!/home/ben/anaconda3/bin/python
from tests.setup_maplike import setup_maplike
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from os.path import join, abspath

if __name__=='__main__':
    ml, true_params = setup_maplike()
    (beta_s_true, beta_d_true, T_d_true) = true_params
    # compute likelihood on grid of parameters
    gen = ml.split_data()
    params_dicts = []
    # generate grid
    nsamp = 32
    beta_d = np.linspace(-0.1, 0.1, nsamp) + beta_d_true
    T_d = np.linspace(-2, 2, nsamp) + T_d_true
    beta_s = np.linspace(-0.1, 0.1, nsamp) + beta_s_true
    # cycle through data one big pixel at a time
    for (mean, var), ipix_spec in zip(gen, range(hp.nside2npix(ml.nside_spec))):
        print("Calculating likelihood for pixel: ", ipix_spec)
        lkl = np.zeros((nsamp, nsamp, nsamp))
        for i, b_d in enumerate(beta_d):
            for j, T in enumerate(T_d):
                for k, b_s in enumerate(beta_s):
                    params = (b_s, b_d, T)
                    lkl[i, j, k] = ml.marginal_spectral_likelihood(params, mean, var)

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
    
