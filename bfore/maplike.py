from __future__ import absolute_import, print_function
import numpy as np
import healpy as hp
from copy import deepcopy
from .skymodel import SkyModel
from .instrumentmodel import InstrumentModel
from scipy import stats, linalg

class MapLike(object) :
    """
    Map-based likelihood

    NOTES:
        - Implement a chi squared function to take in spectral parameters and
        check the full likelihood for amplitudes
    """
    def __init__(self, config_dict, sky_model,instrument_model) :
        """
        Initializes likelihood

        Parameters
        ----------
        sky_model: SkyModel
            SkyModel object describing all sky components, contains the
            SkyModel.fnu method, which is used to calculate the SED.

        instrument_model: InstrumentModel
            InstrumentModel object describing the instrument's response to the
            sky.

        config_dict: dictionary
            Dictionary containing all the setup information for the likelihood.
            This contains frequencies of the data, the data mean, the data
            variance, and which spectral parameters to sample.
            Fields this dictionary must have are:
            - data: data 2 or 3-D array [N_pol,N_pix,N_freq]
            - noisevar: noise variance of the data [N_pol,N_pix,N_freq]
            - var_pars: which parameters to vary (list(str)).
            - fixed_pars: which parameters are fixed (dictionary with fixed values)
            - var_prior_mean: array with the mean value of the prior for each
                 parameter. This value will also be used to initialize any
                 sampler/minimizer.
            - var_prior_width: array with the width of the prior for each parameter.
                 Gaussian priors are implied throughout.
        """
        self.sky = sky_model
        self.inst = instrument_model
        self.__dict__.update(config_dict)
        self.check_parameters()
        if ((self.inst.n_channels!=self.data.shape[-1]) or
            (self.inst.n_channels!=self.noisevar.shape[-1])) :
            raise ValueError("Data does not conform to instrument parameters")
        if self.data.ndim==3 :
            shp=self.data.shape
            self.n_pol=shp[0]
            self.data=self.data.reshape([shp[0]*shp[1],shp[2]]) #Flatten first two dimensions
            self.noisevar=self.noisevar.reshape([shp[0]*shp[1],shp[2]])
        else :
            self.n_pol=1
        self.noiseivar=1./self.noisevar #Inverse variance
        self.dataivar=self.data*self.noiseivar #Inverse variance-weighted data
        self.npix=len(self.data)
        self.var_prior_mean=np.array(self.var_prior_mean)
        self.var_prior_width=np.array(self.var_prior_width)
        self.var_prior_iwidth=1./self.var_prior_width
        # numer of degrees of freedom
        n_amps = self.npix * self.sky.ncomps
        n_spec = len(self.var_pars)
        n_data = self.npix * self.inst.n_channels
        self.dof = float(n_data - n_amps - n_spec)

    def check_parameters(self):
        """ Method to check that all the parameters required by the skymodel
        have been specified as either fixed, with a specific value, or are
        desginated as variable.

        Raises
        ------
        ConfigError
        """
        if any([par in self.fixed_pars for par in self.var_pars]):
            print("Check parameter not in both fixed and variable parameters.")
            exit()
        # get parameters required by SkyModel
        model_pars = set(self.sky.get_param_names())
        # get fixed parameter names, specified in the config
        config_pars = set(self.fixed_pars)
        # update this with the variable parameter names, specified in the config
        config_pars.update(set(self.var_pars))
        # check these sets are the same
        if not (model_pars == config_pars):
            print("Parameter mismatch between model and MapLike configuration")
            exit()
        return

    def f_matrix(self, var_pars_list, inst_params=None) :
        """
        Returns the instrument's response to each of the sky components in each
        frequency channel.

        Parameters
        ----------
        var_pars_list: list
            Parameters necessary to describe all components in the sky model
        inst_params: dictf(x, df, loc=0,
            Parameters describing the instrument (none needed/implemented yet).

        Returns::
        -------
        array_like(float)
            The returned array has shape (N_comp, N_freq).
        """
        # put the list of parameter values into a dictionary
        spec_params = {par_name:par_val for par_name, par_val in zip(self.var_pars, var_pars_list)}
        # add the parameters that are fixed
        spec_params.update(self.fixed_pars)
        return self.inst.convolve_sed(self.sky.fnu,args=spec_params) #TODO: check/optimize

    def get_amplitude_covariance(self, spec_params,
                                 inst_params=None, f_matrix=None):
        """
        Computes the covariance of the different component amplitudes.

        Parameters
        ----------
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
            Array with dimensions (N_pix,N_comp,N_comp), containing the
            noise covariance of all component amplitudes in each pixel and
            polarization channel.
        """
        if f_matrix is None:
            f_matrix = self.f_matrix(spec_params, inst_params)
        # f_matrix -> (N_comp,N_freq)
        fprod=f_matrix[:,None,:]*f_matrix[None,:,:]
        # fprod -> (N_comp,N_comp,N_freq)
        # noiseivar -> (N_pix,N_freq)
        # Output -> (N_pix,N_comp,N_comp)
        return np.sum(fprod[None,:,:,:]*self.noiseivar[:,None,None,:],axis=3) #TODO: check/optimize

    def get_amplitude_mean(self, spec_params,
                            inst_params=None, f_matrix=None, nt_inv_matrix=None):
        """
        Computes the best-fit amplitudes for all components.

        Parameters
        ----------
        spec_params: dict
            Parameters necessary to describe all components in the sky model
        inst_params: dict
            Parameters describing the instrument (none needed/implemented yet).
        f_matrix: array_like(float)
            Array with shape (N_comp, N_freq) (see f_matrix above). If not None,
            the F matrix won't be recalculated.
        nt_matrix: array_like(float)
            Array with shape (N_pix, N_comp, N_comp) (see
            `get_amplitude_covariance` above). If not None, the N_T matrix won't
            be recalculated.

        Returns
        -------
        array_like
            Array with dimensions (N_pix, N_comp).
        """
        # Again, we're allowing F and N_T to be passed to avoid extra operations.
        # Should we be passing choleskys here?
        if f_matrix is None:
            f_matrix = self.f_matrix(spec_params, inst_params)
        # f_matrix -> (N_comp,N_freq)
        if nt_inv_matrix is None:
            nt_inv_matrix = self.get_amplitude_covariance(spec_params, inst_params=inst_params,
                                                          f_matrix=f_matrix)
        # nt_inv_matrix -> (N_pix,N_comp,N_comp)
        y = np.sum(f_matrix[None,:,:]*self.dataivar[:,None,:],axis=2) #TODO: check/optimize
        # y -> (N_pix,N_comp)
        return np.linalg.solve(nt_inv_matrix, y) #TODO: check/optimize

    def logprior(self,spec_params,inst_params=None):
        """ Function to calculate the prior for spectral parameters

        Parameters
        ----------
        spec_params: list
            List of the variable parameters that will be sampled. These must be
            passed in the order of the list self.var_pars.
        inst_params: dict
            Parameters describing the instrument (none needed/implemented yet).

        Returns
        -------
        float
            Prior at this point in parameter space.
        """
        return -np.sum(((spec_params-self.var_prior_mean)*self.var_prior_iwidth)**2)

    def marginal_spectral_likelihood(self, spec_params,
                                     inst_params=None, volume_prior=True,
                                     add_prior=False):
        """ Function to calculate the likelihood marginalized over amplitude
        parameters.

        Parameters
        ----------
        spec_params: list
            List of the variable parameters that will be sampled. These must be
            passed in the order of the list self.var_pars.
        inst_params: dict
            Parameters describing the instrument (none needed/implemented yet).
        add_prior: set to True if you want to include the parameter prior

        Returns
        -------
        float
            Likelihood at this point in parameter space.
        """
        # calculate sed for proposal spectral parameters
        f_matrix = self.f_matrix(spec_params, inst_params=inst_params)
        # f_matrix -> (N_comp,N_freq)
        # get amplitude covariance for proposal spectral parameters
        amp_covar_matrix = self.get_amplitude_covariance(spec_params, inst_params, f_matrix)
        # amp_covar_matrix -> (N_pix,N_comp,N_comp)
        # get amplitude mean for proposal spectral parameters
        amp_mean = self.get_amplitude_mean(spec_params,
                                           inst_params, f_matrix=f_matrix,
                                           nt_inv_matrix=amp_covar_matrix)
        # amp_mean -> (N_pix,N_comp)
        # NOTE:
        #   - einsum only optimized for two matrices contraction, so split this
        #   - This could be also written as (N^-1*d)^T*F^T*amp_mean, which might save some time
        like=np.einsum("ij,ijk,ik->", amp_mean, amp_covar_matrix, amp_mean) #TODO: check/optimize

        if add_prior :
            return like+self.logprior(spec_params,inst_params)
        else :
            return like

    def chi2(self, spec_params, inst_params=None,
             f_matrix=None, volume_prior=True, lnprior=None):
        """ Function to calculate the chi2 of a given set of spectral
        parameters.

        This function first computes the mean amplitude templates, and then uses
        these in the unmarginalized likelihood to compute the chi2 defined by:

        ..math::
            \chi^2 = (d-FT)^T N^{-1}(d-FT)

        Parameters
        ----------
        spec_params: list
            List of the variable parameters that will be sampled. These must be
            passed in the order of the list self.var_pars.
        inst_params: dict
            Parameters describing the instrument (none needed/implemented yet).

        Returns
        -------
        float
            Chi squared for given spectral parameters.
        """
        if f_matrix is None:
            f_matrix = self.f_matrix(spec_params, inst_params)
        # calculate amplitude templates for given spectral parameters
        amp_mean = self.get_amplitude_mean(spec_params,
                                           inst_params=None, f_matrix=None,
                                           nt_inv_matrix=None)
        res=self.data-np.dot(amp_mean,f_matrix) #TODO: check/optimize
        return np.sum(res**2*self.noiseivar)

    def chi2perdof(self, spec_params, inst_params=None,
                f_matrix=None, volume_prior=True, lnprior=None):
        """ Function to calculate the reduced chi2 of a given set of spectral
        parameters.

        This function first computes the mean amplitude templates, and then uses
        these in the unmarginalized likelihood to compute the chi2 defined by:

        ..math::
            \chi^2 = (d-FT)^T N^{-1}(d-FT) / {\rm dof}

        Parameters
        ----------
        spec_params: list
            List of the variable parameters that will be sampled. These must be
            passed in the order of the list self.var_pars.
        inst_params: dict
            Parameters describing the instrument (none needed/implemented yet).

        Returns
        -------
        float
            Chi squared per degree of freedom for given spectral parameters.
        """
        chi2 = self.chi2(spec_params, inst_params=None,
                         f_matrix=f_matrix, volume_prior=volume_prior,
                         lnprior=lnprior)
        return chi2 / float(self.dof)

    def pval(self, spec_params, inst_params=None,
             f_matrix=None, volume_prior=True, lnprior=None):
        """ Function to calculate the p-value of a given set of spectral
        parameters.

        Parameters
        ----------
        spec_params: list
            List of the variable parameters that will be sampled. These must be
            passed in the order of the list self.var_pars.
        inst_params: dict
            Parameters describing the instrument (none needed/implemented yet).

        Returns
        -------
        float
            p-value for given spectral parameters.
        """
        chi2 = self.chi2(spec_params, inst_params=None,
                         f_matrix=f_matrix, volume_prior=volume_prior,
                         lnprior=lnprior)
        return 1. - stats.chi2.cdf(chi2, self.dof)
