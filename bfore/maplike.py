from __future__ import absolute_import, print_function
import numpy as np
import healpy as hp
from copy import deepcopy
from .sky_model import SkyModel
from .instrument_model import InstrumentModel
import numpy as np

class MapLike(object) :
    """
    Map-based likelihood

    NOTES:
        - Need function to break up input data into large spectral parameter
        pixels, and distribute to sampler one at a time, or across tasks.
        - Decide how to pass spectral parameters between maplike and skymodel
        and component classes. One dictionary with all parameters named
        differently would be the easiest, but perhaps confusing. 

    """
    def __init__(self, config_dict, sky_model):#,instrument_model) :
        """
        Initializes likelihood

        Parameters
        ----------
        config_dict: dictionary
            Dictionary containing all the setup information for the likelihood.
            This contains frequencies of the data, the data mean, the data
            variance, and which spectral parameters to sample.
            Fields this dictionary must have are:
            - fpaths_mean: paths to data means (list(str)).
            - fpaths_vars: paths to data variances (list(str)).
            - var_pars: which parameters to vary (list(str)).
            - nus: frequencies (list(float)).

        sky_model: SkyModel
            SkyModel object describing all sky components, contains the
            SkyModel.fnu method, which is used to calculate the SED.
        instrument_model: InstrumentModel
            InstrumentModel object describing the instrument's response to the
            sky.
        """
        self.__dict__.update(config_dict)
        self.sky=sky_model
        #elf.inst=instrument_model
        self.read_data()
        #Here we could precompute the F matrix and interpolate it over spectral parameters.

    def read_data(self):
        """ Method to read input data.
        """
        self.data_mean = read_hpix_maps(self.fpaths_mean)
        self.data_vars = read_hpix_maps(self.fpaths_vars)
        return

    def proposal_spec_params(self, spec_params, sigma=0.1):
        """ Function to calculate a set of proposal spectral parameters for the
        next iteration of the sampler.

        Parameters
        ----------
        spec_params: dictionary
            Dictionary containing the current spectral parameters.
        sigma: float
            Scaling parameter controlling the range of proposal values.

        Returns
        -------
        dictionary
            Dictionary containing the proposal spectral parameters.
        """
        prop = deepcopy(spec_params)
        for par_name in self.var_pars:
            prop[par_name] += sigma * np.random.randn()
        return prop

    def f_matrix(self, spec_params, inst_params=None) :
        """
        Returns the instrument's response to each of the sky components in each frequency channel.
        The returned array has shape (N_pol, N_comp, N_freq).
        spec_params : dictionary of parameters necessary to describe all components in the sky model
        inst_params : dictionary of parameters describing the instrument (none needed/implemented yet).
        """
        #return self.inst.convolve_sed(self.sky.fnu,args=spec_params,instpar=inst_params)
        return self.sky.fnu(self.nus, spec_params)

    def get_amplitude_covariance(self, n_ivar_map, spec_params, inst_params=None, f_matrix=None):
        """
        Computes the covariance of the different component amplitudes.

        Parameters
        ----------
        n_ivar_map: array_like(float)
            2D array with dimensions (N_freq, N_pol,N_pix), where N_pol is the
            number of polarization channels and N_pix is the number of pixels.
            Each element of this array should contain the inverse noise variance
            in that pixel and frequency channel. Uncorrelated noise is assumed.
        spec_params: dict
            Parameters necessary to describe all components in the sky model
        inst_params: dict
            Parameters describing the instrument (none needed/implemented yet).
        f_matrix: array_like(float)
            Array with shape (N_comp, N_freq) (see f_matrix above). If not None,
            the F matrix won't be recalculated.

        Returns
        -------
        array_like(float)
            Array with dimensions (N_pol,N_pix,N_comp,N_comp), containing the
            noise covariance of all component amplitudes in each pixel and
            polarization channel.
        """
        #Note: we should think about where it'd be better to compute/store the Cholesky decomposition of N_T (and even if it'd be better to store the Cholesky of N_T^{-1}).
        if f_matrix is None:
            f_matrix = self.f_matrix(spec_params, inst_params)
        # (N_pol, N_comp, N_freq) x (N_pol, N_comp, N_freq) = (N_pol, N_comp, N_comp, N_freq)
        f_mat_outer = np.einsum("ijk,ilk->ijlk", f_matrix, f_matrix)
        # (N_pol, N_comp, N_comp, N_freq) * (N_freq, N_pol, N_pix) = (N_pol, N_pix, N_comp, N_comp)
        amp_covar_inv = np.einsum("ijkl,min->injk", f_mat_outer, n_ivar_map)
        # Get lower triangular cholesky decomposition (this automatically uses the last two dimensions).
        #lower_cholesky = np.linalg.cholesky(amp_covar_inv)
        # Note: do we need the cholesky decomposition if the amplitudes are not sampled?
        return amp_covar_inv

    def get_amplitude_mean(self,d_map,n_ivar_map,spec_params,inst_params,f_matrix=None,nt_inv_matrix=None) :
        """
        Computes the best-fit amplitudes for all components.

        Parameters
        ----------
        d_map: array_like(float)
            2D array with dimensions (N_pol,N_pix), where N_pol is the number of
            polarization channels and N_pix is the number of pixels. Each
            element of this array should contain the measured
            temperature/polarization in that pixel and frequency channel.
        n_ivar_map: array_like(float)
            2D array with dimensions (N_freq, N_pol,N_pix), where N_pol is the
            number of polarization channels and N_pix is the number of pixels.
            Each element of this array should contain the inverse noise variance
            in that pixel and frequency channel. Uncorrelated noise is assumed.
        spec_params: dict
            Parameters necessary to describe all components in the sky model
        inst_params: dict
            Parameters describing the instrument (none needed/implemented yet).
        f_matrix: array_like(float)
            Array with shape (N_comp, N_freq) (see f_matrix above). If not None,
            the F matrix won't be recalculated.
        nt_matrix: array_like(float)
            Array with shape (N_pol,N_pix,N_comp,N_comp) (see
            `get_amplitude_covariance` above). If not None, the N_T matrix won't
            be recalculated.

        Returns
        -------
        array_like
            Array with dimensions (N_pol,N_pix,N_comp).
        """
        # Again, we're allowing F and N_T to be passed to avoid extra operations.
        # Should we be passing choleskys here?
        if f_matrix is None:
            f_matrix = self.f_matrix(spec_params, inst_params)
        if nt_matrix is None:
            nt_inv_matrix = self.get_amplitude_covariance(n_ivar_map, spec_params, inst_params=inst_params, f_matrix=f_matrix_params)
        # NOTE: n_ivar_map * d_map should not be calculated for each iteration
        # (N_pol, N_comp, N_freq) * (N_freq, N_pol, N_pix) * (N_freq, N_pol, N_pix) = (N_pol, N_pix, N_comp)
        y = np.einsum("ijk,kil,kim->ilj", f_matrix, n_ivar_map, d_map)
        # Get the solution to: N_T_inv T_bar = F^T N^-1 d
        amp_mean = np.linalg.solve(nt_inv_matrix, y)
        return amp_mean

    def marginal_spectral_likelihood(self, d_map, n_ivar_map, spec_params, inst_params=None, volume_prior=True, lnprior=None):
        """ Function to calculate the likelihood marginalized over amplitude
        parameters.

        Parameters
        ----------
        d_map, n_ivar_map: array_like(float)
            Subset of input data pixel mean and pixel variance respectively.
            Only contains pixels within the large pixel over which spectral
            parameters are constant. Shape (Nfreq, Npol, Npix) where
            Npix = (Nside_small / Nside_big) ** 2.
        spec_params: dict
            Parameters necessary to describe all components in the sky model
        inst_params: dict
            Parameters describing the instrument (none needed/implemented yet).

        Returns
        -------
        float
            Likelihood at this point in parameter space.
        """
        # calculate sed for proposal spectral parameters
        f_matrix = self.f_matrix(spec_params, inst_params=inst_params)
        # get amplitude covariance for proposal spectral parameters
        amp_covar_matrix = self.get_amplitude_covariance(n_ivar_map, spec_params, inst_params, f_mat)
        # get amplitude mean for proposal spectral parameters
        amp_mean = self.get_amplitude_mean(d_map, n_ivar_map, spec_params, inst_params, f_matrix=f_matrix,nt_inv_matrix=amp_covar_matrix)
        # NOTE: Need to add option to not cancel the volume prior, and option to add
        # priors on the spectral indices.
        return np.einsum("ijk,ijkl,ijl->", amp_mean, amp_covar_matrix, amp_mean)

    def sample_marginal_spectral_likelihood(self, n_iter):
        """
        Computes the marginal likelihood of the non-amplitude parameters,
        marginalized over the amplitude parameters for a given large pixel in
        which spectral parameters are spatially constant.

        Parameters
        ----------
        d_map, n_ivar_map: array_like(float)
            Subset of input data pixel mean and pixel variance respectively.
            Only contains pixels within the large pixel over which spectral
            parameters are constant. Shape (Nfreq, Npol, Npix) where
            Npix = (Nside_small / Nside_big) ** 2.
        spec_params: dict
            Parameters necessary to describe all components in the sky model
        inst_params: dict
            Parameters describing the instrument (none needed/implemented yet).

        """
        #NOTE: add something in this function or somewhere else that splits up
        # the pixels between tasks such that each task does not need a copy of
        # the whole set of maps.
        chain = []
        loglkl = self.marginal_spectral_likelihood(d_map, n_ivar_map, self.initial_param_guess, inst_params=None, volume_prior=True, lnprior=None)
        chain.append(self.initial_param_guess)
        for i in range(n_iter):
            # proposal set of spectral parameters
            spec_params_prop = self.proposal_spec_params()
            # calculate likelihood at proposal point
            loglkl_prop = self.marginal_spectral_likelihood(d_map, n_ivar_map, spec_params_prop, inst_params=None, volume_prior=True, lnprior=None)
            # calculate acceptance ratio
            acceptance_ratio = np.exp(loglkl_prop - loglkl)
            if acceptance_ratio > np.random.uniform(0, 1):
                spec_params = np.copy(spec_params_prop)
            chain.append(spec_params)
        return chain

def read_hpix_maps(fpaths, verbose=False):
    """ Convenience function for reading in a list of paths to healpix maps and
    returning an array of the maps.

    Parameters
    ----------
    fpaths: list(str)
        List of paths to healpix maps.

    Returns
    -------
    array_like(floats)
        Array of shape (len(`fpaths`), npix) for just T maps, or
        (len(`fpaths`), 3, npix) for TQU maps.
    """
    gen = (hp.read_map(fpath, verbose=verbose, field=(0, 1, 2)) for fpath in fpaths)
    return np.array(list(gen))
