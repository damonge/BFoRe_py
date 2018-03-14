import numpy as np
import emcee
from scipy.optimize import minimize

def run_minimize(func,pos0,dpos=None,method='Powell',tol=None,callback=None,options=None,verbose=False):
    """ Function to find maximum-likelihood parameters

    Parameters
    ----------
    func: function
        Likelihood function to maximize. Must be of the form f(x,*args)
    pos: list(float)
        A best guess of the parameter values to initiate the sampler.
    dpos: list(float)
        Irrelevant (optional)
    method: string
        Minimizer method, only 'Powell' tested so far (optional, default='Powell')
    callback : callable
        Called after each iteration of the minimizer, as callback(xk), where xk
        is the current parameter vector (optional, default=None)
    tol : float
        Tolerance for termination. For detailed control, use solver-specific options.
    options : dict
        Optional dictionary containing additional options for each method. One particularly
        useful option is maxiter:int

    Returns
    -------
    array_like
        array of 2 elements. The first one contains the results of the maximization, the second
        one says whether the maximization was successful.
    """
    if verbose :
        print("Minimizing")
    def mfunc(p,*a) :
        return -func(p,*a)
    res=minimize(mfunc,pos0,method=method,tol=tol,callback=callback,options=options)
    if verbose :
        print("Done minimizing")
    return np.array([res.x,res.success])

def run_emcee(func, pos0, dpos=None, nwalkers=100, nsamps=500,
              nburn=50, verbose=False):
    """ Function to run the emcee sampler on a given likelihood function.

    Parameters
    ----------
    func: function
        Likelihood function to sample, this must obey the requirements of the
        `emcee` module.
    pos: list(float)
        A best guess of the parameter values to initiate the sampler.
    dpos: list(float)
        A best guess of the parameter uncertainties to guide the initial steps
        (optional,default=None).
    nwalkers: int
        Number of independent walkers to start (optional, default=100).
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
    ndim=len(pos0)

    #Set up initial displacement amplitudes
    if dpos is None:
        dpos=1e-2*np.ones(ndim)
    dp=np.zeros(ndim)
    for i,d in enumerate(dpos) :
        if ((d is None) or (d<=0)) :
            dp[i]=1e-2
        else :
            dp[i]=d*0.1
    # initial positions of the walkers
    pos = [pos0 + dp * np.random.randn(ndim) for i in range(nwalkers)]
    # initiate emcee sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, func)
    sampler.run_mcmc(pos, nsamps)
    samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
    if verbose:
        print("Finished sampling.")
    return samples

def clean_pixels(maplike,sampler,d_params=None,**sampler_args):
    """ Function to combine a given MapLike likelihood object and a given
    sampler.

    Parameters
    ----------
    maplike: MapLike
        Instance of the Maplike method.
    sampler: function
        Which sampling function to be used.
    d_params: (list(float))
        Expected width for each parameter (pass None if no idea).
    sampler_args: dict
        Keyword arguments containing hyperparameters specific to whichever
        sampler was chosen.

    Returns
    -------
    list(array_like(float))
        List of MCMC chains corresponding to the pixels in `ipix`.
    """
    samples=sampler(maplike.marginal_spectral_likelihood,
                    pos0=maplike.var_prior_mean,
                    dpos=d_params,
                    **sampler_args)
    return samples
