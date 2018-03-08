from __future__ import print_function
import numpy as np

""" This is just a suggestion for how this submodule might look. Any additional
SEDs or components that can be parameterized by a simple function of nu and
parameters can be added as a function.

We pass parameters between the MapLike, SkyModel, and Component, classes as a
single dictionary, with all the parameters as keywords. So all SED functions
receive all of the parameters. Therefore, parameters in different functions
must use different names. This is not ideal, and should probably be modified,
but was the easiest thing right now.
"""

class Component(object) :
    """
    Empty component class
    No parameters

    NOTE:
        - If the models we are using remain at this level of complexity, I
        don't think having a separate base class makes sense. These functions
        should all just be assembled in SkyModel, in the same way as Component
        is currently doing. This is currently just wrapping individual
        functions, without adding anything.
    """
    def __init__(self, comp_name) :
        """ SED must be in uK_RJ units.
        """
        self.comp_name = comp_name
        self.sed = globals()[comp_name]
        return

    def __call__(self, nu, pars) :
        """ Method to call the SED with whichi a given instance was initialized.

        Parameters
        ----------
        nu: float, or array_like(float)
            Frequency or list of frequencies, in GHz, at which to evaluate the
            SED.
        pars:  dict
            Dictionary of parameters taken by the SED.
        """
        return self.sed(nu, **pars)

    def get_description(self):
        print("Component SED name: ", self.comp_name)
        print(self.sed.__doc__, "\n --------------- \n")
        pass


def cmb(nu, *args, **kwargs):
    """ Function to compute CMB SED.

    Parameters
    ----------
    nu: float, or array_like(float)
        Frequency in GHz.
    """
    if isinstance(nu, (float, int)):
        nu = [nu]
    nu = np.array(nu)
    x = 0.0176086761 * nu
    ex = np.exp(x)
    sed = ex * (x / (ex - 1)) ** 2
    return np.concatenate((sed, sed, sed)).reshape((3, -1))

def syncpl(nu, *args, **kwargs):
    """ Function to compute synchrotron power law SED.

    Parameters
    ----------
    nu: float, or array_like(float)
        Frequency in GHz.
    beta_s: float
        Power law index in RJ units.

    Returns
    -------
    array_like(float)
        Synchroton SED relative to reference frequency.
    """
    if isinstance(nu, (float, int)):
        nu = [nu]
    nu = np.array(nu)
    x = nu / kwargs['nu_ref_s']
    sed = x ** kwargs['beta_s']
    return np.concatenate((sed, sed, sed)).reshape((3, -1))


def dustmbb(nu, * args, **kwargs):
    """ Function to compute modified blackbody dust SED.

    Parameters
    ----------
    nu: float or array_like(float)
        Freuency at which to calculate SED.
    nu_ref_d: float
        Reference frequency in GHz.
    beta_d: float
        Power law index of dust opacity.
    T_d: float
        Temperature of the dust.

    Returns
    -------
    array_like(float)
        SED of dust modified black body relative to reference frequency.
    """
    if isinstance(nu, (float, int)):
        nu = [nu]
    nu = np.array(nu)
    x_to = 0.0479924466 * nu / kwargs['T_d']
    x_from = 0.0479924466 * kwargs['nu_ref_d'] / kwargs['T_d']
    sed = (nu / kwargs['nu_ref_d']) ** (1 + kwargs['beta_d']) * (np.exp(x_from) - 1) / (np.exp(x_to) - 1)
    return np.concatenate((sed, sed, sed)).reshape((3, -1))
