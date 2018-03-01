from bfore import MapLike
import numpy as np
import healpy as hp
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
    ml = MapLike(config_dict)
    print(ml.nus)
    print(ml.fpaths_mean)
    print(ml.fpaths_vars)
    print(ml.data_mean.shape)
    print(ml.data_vars.shape)
    return

if __name__=="__main__":
    config_dict = test_setup()
    test_maplike(config_dict)
