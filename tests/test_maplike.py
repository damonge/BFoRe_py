from __future__ import absolute_import
from unittest import TestCase
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from .setup_maplike import setup_maplike


class test_MapLike(TestCase):
    def setUp(self):
        self.maplike, self.true_params = setup_maplike()
        return

    def test_array_shapes(self):
        covar_shape = self.maplike.get_amplitude_covariance(self.true_params).shape
        print(covar_shape)
        return

    def test_likelihood(self):
        # get input data for each large pixel
        (beta_s_true, beta_d_true, T_d_true, beta_c_true) = self.true_params
        # generate grid of parameters
        nsamp = 16
        beta_d = np.linspace(-0.1, 0.1, nsamp) + beta_d_true
        T_d = np.linspace(-2, 2, nsamp) + T_d_true
        beta_s = np.linspace(-0.1, 0.1, nsamp) + beta_s_true
        beta_c = np.linspace(-0.05, 0.05, nsamp) + beta_c_true
        lkl = np.zeros((nsamp, nsamp, nsamp, nsamp))
        params_dicts = []
        for i, b_d in enumerate(beta_d):
            print(i)
            for j, T in enumerate(T_d):
                for k, b_s in enumerate(beta_s):
                    for l, b_c in enumerate(beta_c):
                        params = (b_s, b_d, T, b_c)
                        lkl[i, j, k, l] = self.maplike.marginal_spectral_likelihood(params)

        # plot 2d posteriors
        X, Y = np.meshgrid(T_d, beta_s)
        plt.title(r"$T_d - \beta_s$")
        plt.xlabel(r"$T_d$")
        plt.ylabel(r"$\beta_s$")
        plt.contourf(X, Y, np.sum(lkl, axis=(0, 3)))
        plt.axvline(x=T_d_true, linestyle="--")
        plt.axhline(y=beta_s_true, linestyle="--")
        plt.colorbar(label=r"$F^T N_T^{-1} F$")
        plt.show()

        X, Y = np.meshgrid(beta_d, beta_s)
        plt.contourf(X, Y, np.sum(lkl, axis=(1, 3)))
        plt.axvline(x=beta_d_true, linestyle="--")
        plt.axhline(y=beta_s_true, linestyle="--")
        plt.title(r"$\beta_d - \beta_s$")
        plt.xlabel(r"$\beta_d$")
        plt.ylabel(r"$\beta_s$")
        plt.colorbar(label=r"$F^T N_T^{-1} F$")
        plt.show()

        X, Y = np.meshgrid(beta_d, T_d)
        plt.contourf(X, Y, np.sum(lkl, axis=(2, 3)))
        plt.axvline(x=beta_d_true, linestyle="--")
        plt.axhline(y=T_d_true, linestyle="--")
        plt.title(r"$\beta_d - T_d$")
        plt.xlabel(r"$\beta_d$")
        plt.ylabel(r"$T_d$")
        plt.colorbar(label=r"$F^T N_T^{-1} F$")
        plt.show()

        X, Y = np.meshgrid(beta_s, beta_c)
        plt.contourf(X, Y, np.sum(lkl, axis=(0, 1)))
        plt.axvline(x=beta_s_true, linestyle="--")
        plt.axhline(y=beta_c_true, linestyle="--")
        plt.title(r"$\beta_s - \beta_c$s")
        plt.xlabel(r"$\beta_s$")
        plt.ylabel(r"$\beta_c$")
        plt.colorbar(label=r"$F^T N_T^{-1} F$")
        plt.show()

        # plot 1d posteriors
        beta_s_1d = np.sum(lkl, axis=(0, 1, 3))
        T_d_1d = np.sum(lkl, axis=(0, 2, 3))
        beta_d_1d = np.sum(lkl, axis=(1, 2, 3))
        beta_c_1d = np.sum(lkl, axis=(0, 1, 2))

        plt.plot(beta_s, beta_s_1d)
        plt.axvline(beta_s_true, color='k', linestyle='--')
        plt.title(r"1D marg posterior for $\beta_s$, max={:f}".format(beta_s[np.argmax(beta_s_1d)]))
        plt.show()

        plt.plot(T_d, T_d_1d)
        plt.axvline(T_d_true, color='k', linestyle='--')
        plt.title("1D marg posterior for $T_d$, max={:f}".format(T_d[np.argmax(T_d_1d)]))
        plt.show()
        
        plt.plot(beta_d, beta_d_1d)
        plt.axvline(beta_d_true, color='k', linestyle='--')
        plt.title(r"1D marg posterior for $\beta_d$, max={:f}".format(beta_d[np.argmax(beta_d_1d)]))
        plt.show()
        
        plt.plot(beta_c, beta_c_1d)
        plt.axvline(beta_c_true, color='k', linestyle='--')
        plt.title(r"1D marg posterior for $\beta_c$, max={:f}".format(beta_c[np.argmax(beta_c_1d)]))
        plt.show()

        return
