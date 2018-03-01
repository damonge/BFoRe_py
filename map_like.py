import numpy as np

class MapLike(object) :
    """
    Map-based likelihood
    """

    def __init__(self,sky_model,instrument_model) :
        """
        Initializes likelihood
        sky_model (SkyModel) : SkyModel object describing all sky components
        instrument_model (InstrumentModel) : object describing the instrument's response to the sky
        """
        self.sky=sky_model
        self.inst=instrument_model

        #Here we could precompute the F matrix and interpolate it over spectral parameters.
        
    def f_matrix(self,spec_params,inst_params=None) :
        """
        Returns the instrument's response to each of the sky components in each frequency channel.
        The returned array has shape (N_comp,N_freq).
        spec_params : dictionary of parameters necessary to describe all components in the sky model
        inst_params : dictionary of parameters describing the instrument (none needed/implemented yet).
        """
        return self.inst.convolve_sed(self.sky.fnu,args=spec_params,instpar=inst_params)

    def get_amplitude_covariance(self,n_ivar_map,spec_params,inst_params,f_matrix=None) :
        """
        Computes the covariance of the different component amplitudes.
        n_ivar_map (array_like) : 2D array with dimensions (N_pol,N_pix), where N_pol is the number of polarization channels and N_pix is the number of pixels. Each element of this array should contain the inverse noise variance in that pixel and frequency channel. Uncorrelated noise is assumed.
        spec_params : dictionary of parameters necessary to describe all components in the sky model
        inst_params : dictionary of parameters describing the instrument (none needed/implemented yet).
        f_matrix (array_like) : array with shape (N_comp,N_freq) (see f_matrix above). If not None, the F matrix won't be recalculated.
        return : an array with dimensions (N_pol,N_pix,N_comp,N_comp), containing the noise covariance of all component amplitudes in each pixel and polarization channel.
        """
        #Note: we should think about where it'd be better to compute/store the Cholesky decomposition of N_T (and even if it'd be better to store the Cholesky of N_T^{-1}).

    def get_amplitude_mean(self,d_map,n_ivar_map,spec_params,inst_params,f_matrix=None,nt_matrix=None) :
        """
        Computes the best-fit amplitudes for all components.
        d_map (array_like) : 2D array with dimensions (N_pol,N_pix), where N_pol is the number of polarization channels and N_pix is the number of pixels. Each element of this array should contain the measured temperature/polarization in that pixel and frequency channel.
        n_ivar_map (array_like) : 2D array with dimensions (N_pol,N_pix). Each element of this array should contain the inverse noise variance in that pixel and frequency channel. Uncorrelated noise is assumed.
        spec_params : dictionary of parameters necessary to describe all components in the sky model
        inst_params : dictionary of parameters describing the instrument (none needed/implemented yet).
        f_matrix (array_like) : array with shape (N_comp,N_freq) (see f_matrix above). If not None, the F matrix won't be recalculated.
        nt_matrix (array_like) : array with shape (N_pol,N_pix,N_comp,N_comp) (see get_amplitude_covariance above). If not None, the N_T matrix won't be recalculated.
        return : an array with dimensions (N_pol,N_pix,N_comp), containing the best-fit noise covariance of all component amplitudes in each pixel and polarization channel.
        """
        #Again, we're allowing F and N_T to be passed to avoid extra operations. Should we be passing choleskys here?

    def marginal_spectral_likelihood(self,d_map,n_ivar_map,spec_params,inst_params) :
        """
        Computes the marginal likelihood of the non-amplitude parameters marginalized over the amplitude parameters.
        """
        #Again, we should be smart about making this function as fast as possible (in terms of choleskys etc.

