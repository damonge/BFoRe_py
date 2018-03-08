import numpy as np
from .components import Component

class SkyModel(object) :
    """
    Sky model. Basically defined by a
    set of components and some extra parameters (e.g. polarization/temp)
    """
    comps=[]

    def __init__(self, components, is_polarized=False, extra_components=None):
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
        self.components = [Component(comp_name) for comp_name in components]
        return

    def get_model_parameters(self):
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
        return np.array([sed(nu, params) for sed in self.components])
