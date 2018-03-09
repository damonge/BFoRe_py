import numpy as np
import emcee
from schwimmbad import MultiPool
from functools import partial

def run_emcee(data, func, pos0=None, nwalkers=100, ndim=3, nsamps=500, nburn=50,
                verbose=False):
    """ Function to run the emcee sampler on a given set of input data and
    likelihood function.

    Parameters
    ----------
    data: tuple(array_like(float))
        Tuple containing two arrays: the mean and variance of the observations
        within one large spectral parameter pixel. This object is returned
        by the MapLike.split_data() method.
    func: function
        Likelihood function to sample, this must obey the requirements of the
        `emcee` module.
    pos: list(float)
        A best guess of the parameter values to initiate the sampler (optional,
        default=None).
    nwalkers: int
        Number of independent walkers to start (optional, default=100).
    ndim: int
        The number of free parameters to sample this must match the number of
        free parameters given to MapLike (optional, deafault=3).
    nsamps: int
        Number of samples to be taken by each walker (optional, default=500).
    nbrun: int
        Number of samples to be taken as burn-in, and discarded before returning
        the chain.

    Returns
    -------
    array_like(float)
        Chain of samples of shape (nsamps * nwalkers - nburn * nwalkers, ndim).
    """
    if verbose:
        print("Sampling")
    if pos0 is None:
        pos0 = np.zeros(ndim)
    # initial positions of the walkers
    pos = [pos0 + 1e-2 * np.random.randn(ndim) for i in range(nwalkers)]
    # unpack input data
    (mean, var) = data
    # initiate emcee sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, func, args=(mean, var))
    sampler.run_mcmc(pos, nsamps)
    samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
    if verbose:
        print("Finished sampling.")
    return samples

def clean_pixels(maplike, sampler, ipix=None, **sampler_args):
    """ Function to combine a given MapLike likelihood object and a given
    sampler, and distribute tasks across an MPI pool.

    Parameters
    ----------
    maplike: MapLike
        Instance of the Maplike method.
    sampler: function
        Which sampling function to be used.
    ipix: int or list(int)
        List of pixel inidices which are to be sampled. If None sample all
        pixels on the coarser spectral parameter grid.
    sampler_args: dict
        Keyword arguments containing hyperparameters specific to whichever
        sampler was chosen.

    Returns
    -------
    list(array_like(float))
        List of MCMC chains corresponding to the pixels in `ipix`.
    """
    # get the data as a generator which iterates each spectral parameter pixel's
    # corresponding amplitude data.
    gen = maplike.split_data(ipix=ipix)
    # manipulate the MapLike likelihood method to get a function with only one
    # argument (which is required by the map function used to scatter tasks).
    # This argument is a list of the variable spectral parameters.
    loglkl_func = partial(sampler, func=maplike.marginal_spectral_likelihood,
                            **sampler_args)
    # distribute pixels between pool processes.
    with MultiPool() as pool:
        samples_list = list(pool.map(loglkl_func, gen))
    return samples_list
