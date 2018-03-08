from __future__ import absolute_import, print_function
import numpy as np
import healpy as hp
from copy import deepcopy
from .skymodel import SkyModel
from .instrumentmodel import InstrumentModel
import numpy as np

class MapLike(object) :
    """
    Map-based likelihood

    NOTES:
        - Implement a chi squared function to take in spectral parameters and
        check the full likelihood for amplitudes
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
            - nside_spec: nside at which to vary the spectral parameters (int).

        sky_model: SkyModel
            SkyModel object describing all sky components, contains the
            SkyModel.fnu method, which is used to calculate the SED.
        instrument_model: InstrumentModel
            InstrumentModel object describing the instrument's response to the
            sky.
        """
        self.__dict__.update(config_dict)
        self.sky = sky_model
        #self.inst = instrument_model
        self.read_data()
        #Here we could precompute the F matrix and interpolate it over spectral parameters.

    def read_data(self):
        """ Method to read input data. The `self.data_mean` and `self.data_vars`
        will have shape: (N_freqs, N_pol, N_pix)
        """
        self.data_mean = read_hpix_maps(self.fpaths_mean, nest=False)
        self.data_vars = read_hpix_maps(self.fpaths_vars, nest=False)
        self.nside_base = hp.get_nside(self.data_mean[0][0])
        return

    def split_data(self, ipix=None):
        """ Generator function to return the data each task is to work on. This
        is one large spectral index pixel, and the corresponding pixels at the
        base amplitude resolution.

        Parameters
        ----------
        ipix: int, list(int)
            If passed this parameter instructs which pixels to return,
            defined in the nested indexing scheme.  This may be a single
            pixel, or a list of pixels, which will be returned as a
            generator. If this parameter is not passed, all pixels will
            be returned in the form of a generator
            (optional, default=None).

        Returns
        -------
        generator
            Generator function that yields a tuple containing the data mean
            and variance within one large pixel over which the spectral
            paraemters are held constant.
        """
        npix_spec = hp.nside2npix(self.nside_spec)
        npix_base = hp.nside2npix(self.nside_base)
        nside_sub = int(npix_base / npix_spec)

        if ipix is not None:
            if isinstance(ipix, int):
                ipixs = [ipix]
            if isinstance(ipix, list):
                ipixs = ipix
        else:
            ipixs = range(npix_spec)

        for i in ipixs:
            inds = hp.nest2ring(self.nside_base, range(i * nside_sub, (i + 1) * nside_sub))
            mean = self.data_mean[:, :, inds]
            vars = self.data_vars[:, :, inds]
            yield (mean, vars)

    def f_matrix(self, spec_params, inst_params=None) :
        """
        Returns the instrument's response to each of the sky components in each
        frequency channel.

        Parameters
        ----------
        spec_params: dict
            Parameters necessary to describe all components in the sky model
        inst_params: dict
            Parameters describing the instrument (none needed/implemented yet).

        Returns::
        -------
        array_like(float)
            The returned array has shape (N_pol, N_comp, N_freq).
        """
        #return self.inst.convolve_sed(self.sky.fnu,args=spec_params,instpar=inst_params)
        return self.sky.fnu(self.nus, spec_params)

    def get_amplitude_covariance(self, n_ivar_map, spec_params,
                                    inst_params=None, f_matrix=None):
        """
        Computes the covariance of the different component amplitudes.

        Parameters
        ----------
        n_ivar_map: array_like(float)
            2D array with dimensions (N_freq, N_pol, N_pix), where N_pol is the
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
        if f_matrix is None:
            f_matrix = self.f_matrix(spec_params, inst_params)
        # (N_comp, N_pol, N_freq) x (N_comp, N_pol, N_freq) = (N_pol, N_comp, N_comp, N_freq)
        f_mat_outer = np.einsum("ijk,ljk->jilk", f_matrix, f_matrix)
        # (N_pol, N_comp, N_comp, N_freq) * (N_freq, N_pol, N_pix) = (N_pol, N_pix, N_comp, N_comp)
        amp_covar_inv = np.einsum("ijkl,lin->injk", f_mat_outer, n_ivar_map)
        return amp_covar_inv

    def get_amplitude_mean(self, d_map, n_ivar_map, spec_params,
                            inst_params=None, f_matrix=None, nt_inv_matrix=None):
        """
        Computes the best-fit amplitudes for all components.

        Parameters
        ----------
        d_map: array_like(float)
            array with dimensions (N_freq, N_pol, N_pix), where N_pol is the number of
            polarization channels and N_pix is the number of pixels. Each
            element of this array should contain the measured
            temperature/polarization in that pixel and frequency channel.
        n_ivar_map: array_like(float)
            array with dimensions (N_freq, N_pol, N_pix), where N_pol is the
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
            Array with shape (N_pol, N_pix, N_comp, N_comp) (see
            `get_amplitude_covariance` above). If not None, the N_T matrix won't
            be recalculated.

        Returns
        -------
        array_like
            Array with dimensions (N_pol, N_pix, N_comp).
        """
        # Again, we're allowing F and N_T to be passed to avoid extra operations.
        # Should we be passing choleskys here?
        if f_matrix is None:
            f_matrix = self.f_matrix(spec_params, inst_params)
        if nt_inv_matrix is None:
            nt_inv_matrix = self.get_amplitude_covariance(n_ivar_map,
                spec_params, inst_params=inst_params, f_matrix=f_matrix)
        # NOTE: n_ivar_map * d_map should not be calculated for each iteration
        # (N_comp, N_pol, N_freq) * (N_freq, N_pol, N_pix) * (N_freq, N_pol, N_pix) = (N_pol, N_pix, N_comp)
        y = np.einsum("jik,kil,kil->ilj", f_matrix, n_ivar_map, d_map)
        # Get the solution to: N_T_inv T_bar = F^T N^-1 d
        amp_mean = np.linalg.solve(nt_inv_matrix, y)
        return amp_mean

    def marginal_spectral_likelihood(self, spec_params_list, d_map, n_ivar_map,
                                        inst_params=None, volume_prior=True,
                                        lnprior=None):
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
        spec_params = {
            'beta_s': spec_params_list[0],
            'beta_d': spec_params_list[1],
            'T_d': spec_params_list[2],
            'nu_ref_d': 353.,
            'nu_ref_s': 23.
        }
        # calculate sed for proposal spectral parameters
        f_matrix = self.f_matrix(spec_params, inst_params=inst_params)
        # get amplitude covariance for proposal spectral parameters
        amp_covar_matrix = self.get_amplitude_covariance(n_ivar_map,
                                            spec_params, inst_params, f_matrix)
        # get amplitude mean for proposal spectral parameters
        amp_mean = self.get_amplitude_mean(d_map, n_ivar_map, spec_params,
                                            inst_params, f_matrix=f_matrix,
                                            nt_inv_matrix=amp_covar_matrix)
        return np.einsum("ijk,ijkl,ijl->", amp_mean, amp_covar_matrix, amp_mean)


def read_hpix_maps(fpaths, verbose=False, *args, **kwargs):
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

    NOTE:
        - Add option for choosing polarization or not.
    """
    gen = (hp.read_map(fpath, verbose=verbose, field=(0, 1, 2), **kwargs) for fpath in fpaths)
    return np.array(list(gen))
