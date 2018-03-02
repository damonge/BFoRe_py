#!/home/ben/anaconda3/bin/python
from bfore import MapLike, SkyModel
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from os.path import join, abspath

def test_setup():
    nside = 8
    nus = [10., 30., 45., 90., 100., 143., 350., 450.]
    ells = np.linspace(0, 3 * nside, 3 * nside + 1)
    cltt_ref = np.zeros_like(ells)
    cltt_ref[2:] = 5. * (ells[2:] / 80.) ** -3.2
    cltt = lambda nu: cltt_ref * (nu / 23.) ** -3.
    maps = [hp.synfast([cltt(nu), 0.1 * cltt(nu), 0.2 * cltt(nu), 0.3 * cltt(nu)], nside, verbose=False) for nu in nus]
    vars = [0.1 * m for m in maps]
    test_dir = abspath("test_data")
    fpaths_mean = [join(test_dir, "mean_nu{:03d}.fits".format(int(nu))) for nu in nus]
    fpaths_vars = [join(test_dir, "vars_nu{:03d}.fits".format(int(nu))) for nu in nus]

    for nu, m, fm, v, fv in zip(nus, maps, fpaths_mean, vars, fpaths_vars):
        hp.write_map(fm, m, overwrite=True)
        hp.write_map(fv, v, overwrite=True)

    config_dict = {
        "nus": nus,
        "fpaths_mean": fpaths_mean,
        "fpaths_vars": fpaths_vars
    }
    return config_dict

def test_maplike(config_dict):
    skymodel = SkyModel()
    ml = MapLike(config_dict, skymodel)
    print(ml.nus)
    print(ml.fpaths_mean)
    print(ml.fpaths_vars)
    print(ml.data_mean.shape)
    print(ml.data_vars.shape)
    params = {
    "sync_pl": [23., -3.1],
    "dust_mbb": [353., 1.51, 19.],
    }
    print(ml.f_matrix(params))
    print(ml.f_matrix(params))
    return

def test_skymodel():
    skymodel = SkyModel()

    params = {
    "sync_pl": [23., -3.1],
    "dust_mbb": [353., 1.51, 19.],
    }
    freqs = np.logspace(1, 2.5, 100)
    fnus = skymodel.fnu(freqs, params)

    fig, ax = plt.subplots(1, 1)
    ax.loglog(freqs, fnus[0], label='cmb')
    ax.loglog(freqs, fnus[1], label='sync')
    ax.loglog(freqs, fnus[2], label='dust')
    ax.legend()
    plt.show()
    return

if __name__=="__main__":
    #test_skymodel()

    config_dict = test_setup()
    test_maplike(config_dict)
