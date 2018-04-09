import numpy as np
from .components import Component

class SkyModel(object) :
    """
    Sky model. Basically defined by a
    set of components and some extra parameters (e.g. polarization/temp)
    """
    def __init__(self, comp_names, is_polarized=False, extra_components=None):
        """
        Initializes a sky model

        Parameters
        ----------
        components:  list(str)
            List of strings corresponding to the SEDs of the various components.
            These must match names of SED functions available in the
            `bfore.components` submodule.
        is_polarized: bool
            Determines whether maps are I or (Q,U)
        extra_components: list(`Component`)
            List of extra `ComponentBase` objects.
        """
        # Initialize list of requested components.
        self.components = [Component(comp_name) for comp_name in comp_names]
        # get list of all parameter names in the SEDs, excluding frequency `nu`.
        self.sed_param_names= []
        for component in self.components:
            self.sed_param_names += component.get_sed_parameters()
        # get list of all parameter names in the C_ells
        self.cl_param_names= []
        for component in self.components:
            self.cl_param_names += component.get_cl_parameters()
        # get list of lists of parameter names, per component
        # (length = len(components)).
        self.comp_sed_par_names = [comp.get_sed_parameters() for comp in self.components]
        self.comp_cl_par_names = [comp.get_cl_parameters() for comp in self.components]
        self.ncomps = len(self.components)
        return

    def get_sed_param_names(self):
        return self.sed_param_names

    def get_cl_param_names(self):
        return self.cl_param_names

    def get_model_description(self):
        """ Function to collect all the parameters required by the components
        in the model, this just prints the docstring of the SEDs specified
        in the `components` argument initlizing this class.
        """
        print("This instance of SkyModel contains the following SEDs: \n")
        [component.get_description() for component in self.components]
        pass

    def fnu(self, nu, params) :
        """
        Return matrix of SEDs

        Parameters
        ----------
        nu: array_like(float)
            Frequencies in GHz at which to calculate the spectrum.
        params: dict
            Parameters for all the SEDs

        Returns
        -------
        array_like(float)
            Matrix containing the scaling for each parameter. Shape is
            (N_pol, N_comp, N_freq)
        """
        # if nu is not already array_like, make it so
        if not hasattr(nu, '__iter__'):
            nu = [nu]
        # if nu is a list make it an array
        nu = np.array(nu)
        # convert dictionary of parameter names, to a list of tuples containing
        # the arguments for each of the seds.
        component_params = [tuple(params[par_name] for par_name in comp_sed_par_names)
                            for comp_sed_par_names in self.comp_sed_par_names]
        # calculate the seds
        # Returns Ncomp x Nfreq array
        return np.array([comp.spectrum(nu, params) for (comp, params) in zip(self.components, component_params)])
