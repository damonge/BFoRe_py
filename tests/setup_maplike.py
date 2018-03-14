from bfore import MapLike, SkyModel, InstrumentModel
from bfore.components import syncpl, dustmbb, cmb, sync_curvedpl
import tempfile
import numpy as np
import healpy as hp
from os.path import join

def setup_maplike():
    # define true spectral parameters
    beta_s_true = -3.
    beta_d_true = 1.6
    beta_c_true = 0.05
    T_d_true = 20.
    nu_ref_s = 23.
    nu_ref_d = 353.
    nside_spec = 2
    nside = 8
    components = ["sync_curvedpl", "dustmbb", "cmb"]
    nus = [10., 20., 25., 45., 90., 100., 143., 217., 300., 350., 400., 500.]
    sigmas = [1. * sig for sig in [110., 50., 36., 8., 4, 4, 10.1, 20., 25., 30., 40., 50.]]
    #define delta bandpasses
    bps=np.array([{'nu':np.array([n-0.5,n+0.5]),'bps':np.array([1])} for n in nus])
    # generate fake synch and dust as GRFs
    ells = np.linspace(0, 3 * nside, 3 * nside + 1)
    cl_s = np.zeros_like(ells)
    cl_d = np.zeros_like(ells)
    cl_s[2:] = 10. * (ells[2:] / 80.) ** - 3.2
    cl_d[2:] = 10. * (ells[2:] / 80.) ** - 3.2
    # the templates of dust and synchrotron at their reference frequencies
    temp_s = np.array(hp.synfast([cl_s, cl_s, cl_s, cl_s], nside, verbose=False, pol=True))[1:]
    temp_d = np.array(hp.synfast([cl_d, cl_d, cl_d, cl_d], nside, verbose=False, pol=True))[1:]
    temp_c = np.array(hp.synfast([cl_d, cl_d, cl_d, cl_d], nside, verbose=False, pol=True))[1:]
    # the synchrotron and dust signals separates
    synch = np.array([temp_s * sync_curvedpl(np.array([nu]), beta_s=beta_s_true, nu_ref_s=nu_ref_s, beta_c=beta_c_true) for nu in nus])
    dust = np.array([temp_d * dustmbb(np.array([nu]), beta_d=beta_d_true, T_d=T_d_true, nu_ref_d=nu_ref_d) for nu in nus])
    cmbs = np.array([temp_c * cmb(np.array([nu])) for nu in nus])
    # the noise maps
    noise = [add_noise(sig, nside) for sig in sigmas]
    # these are the simulated observations mean and variance
    # synch + dust + noise
    maps = np.transpose(np.array([d + s + c + n  for d, s, c, n in zip(dust, synch, cmbs, noise)]),
                        axes=[1,2,0])
    # inverse pixel noise variance
    vars = np.transpose(np.array([np.ones((2, hp.nside2npix(nside))) * pixel_var(sig, nside) for sig in sigmas]),axes=[1,2,0])

    config_dict = {
        "data": maps,
        "noisevar":vars,
        "fixed_pars": {"nu_ref_d": nu_ref_d, "nu_ref_s": nu_ref_s},
        "var_pars": ["beta_s", "beta_d", "T_d", "beta_c"],
        "var_prior_mean": [beta_s_true,beta_d_true,T_d_true,beta_c_true],
        "var_prior_width": [1.,1.,1.,1.]
        }
    skymodel = SkyModel(components)
    instrumentmodel = InstrumentModel(bps)
    ml=MapLike(config_dict,skymodel,instrumentmodel)
    return ml, (beta_s_true, beta_d_true, T_d_true, beta_c_true)

def pixel_var(sigma_amin, nside):
    """ Function to compute the variance of the noise in each pixel for a given
    noise level in sigma arcminute, and a given nside.

    Parameters
    ----------
    sigma_amin: float
        noise level.
    nside: int
        Nside at which to calculate pixel variance

    Returns
    -------
    float
        Noise variance in each pixel.
    """
    npix = hp.nside2npix(nside)
    amin_sq_per_pix = 4 * np.pi * (180. * 60. / np.pi) ** 2 / npix
    pixel_var = sigma_amin ** 2 / amin_sq_per_pix
    return pixel_var

def add_noise(sigma_amin, nside):
    """ Generate a noise realization for a given noise level and nside.

    Parameters
    ----------
    sigma_amin: float
        noise level.
    nside: int
        Nside at which to calculate pixel variance

    Returns
    -------
    array_like(float)
        Noise realization.
    """
    sigma_pix = np.sqrt(pixel_var(sigma_amin, nside))
    noise = np.random.randn(2, hp.nside2npix(nside)) * sigma_pix
    return noise
