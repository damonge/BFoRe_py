from __future__ import print_function
import numpy as np
import inspect
import sys
from .utils import cl_power_law

class Component(object):
    def __init__(self, comp_name,dict_params={}):
        """ SED must be in uK_RJ units.
        """
        self.comp_name = comp_name
        self.info= globals()[comp_name+'_setup'](dict_params)
        self.sed = globals()[comp_name+'_sed']
        self.cl = globals()[comp_name+'_cl']
        return

    def __call__(self, nu, args):
        """ Method to call the SED with whichi a given instance was initialized.

        Parameters
        ----------
        nu: float, or array_like(float)
            Frequency or list of frequencies, in GHz, at which to evaluate the
            SED.
        args:  tuple
            Tuple containing the positional parameters taken by the SED.
        """
        return self.sed(nu, *args)

    def get_description(self):
        print("Component SED name: ", self.comp_name)
        print(self.sed.__doc__, "\n --------------- \n")
        pass

    def get_sed_parameters(self):
        """ Method to fetch the keywork arguments for the SEDs of various components
        and return them. This is used to build a list of possible parameters
        that may be varied by MapLike.
        """
        if sys.version_info[0]>= 3:
            sig = inspect.signature(self.sed)
            pars = list(sig.parameters.keys())
        else :
            pars = list(inspect.getargspec(self.sed).args)
        return list(filter(lambda par: par not in ['nu', 'args', 'kwargs'], pars))

    def get_cl_parameters(self):
        """ Method to fetch the keywork arguments for the C_ells of various components
        and return them. This is used to build a list of possible parameters
        that may be varied by MapLike.
        """
        if sys.version_info[0]>= 3:
            sig = inspect.signature(self.cl)
            pars = list(sig.parameters.keys())
        else :
            pars = list(inspect.getargspec(self.cl).args)
        return list(filter(lambda par: par not in ['args', 'kwargs','dict_cl',
                                                   'ell','inc_t','inc_e','inc_b'], pars))

    def spectrum(self,nu,args) :
        """ Method to call the SED with whichi a given instance was initialized.

        Parameters
        ----------
        nu: float, or array_like(float)
            Frequency or list of frequencies, in GHz, at which to evaluate the
            SED.
        args:  tuple
            Tuple containing the positional parameters taken by the SED.
        """
        return self.sed(nu, *args)

    def c_ell(self,ell,args,inc_t=False,inc_e=True,inc_b=True) :
        return self.cl(self.info,ell,*args,inc_t=inc_t,inc_e=inc_e,inc_b=inc_b)


def cl_power_law(dict_cl,ell,att,aee,abb,ate,
                 alpha,ell_ref,inc_t=False,inc_e=True,inc_b=True) :
    """ Generic function computing a power-law power spectrum

    Parameters
    ----------
    dict_cl:
        Dictionary containing the power spectrum templates
    ell: array_like
        Multipoles at which you want the power spectrum
    att,aee,abb,ate: float
        Amplitude of the power spectrum (in TT,EE,BB and TE respectively)
    alpha: float
        Spectral tilt (D_l \propto (ell/ell_ref)^(alpha-2))
    ell_ref: float
        Pivot multipole
    inc_t: bool
        Contains temperature?
    inc_e: bool
        Contains E-modes?
    inc_b: bool
        Contains B-modes?
    """
    npol=int(inc_t)+int(inc_e)+int(inc_b)+int(inc_t and inc_e)
    nell=len(ell)
    cl_out=np.zeros([npol,nell])
    ell_template=(ell/ell_ref)**(alpha-2.)
    ind_pol=0
    if inc_t :
        cl_out[ind_pol]=att*ell_template
        ind_pol+=1
    if inc_e :
        cl_out[ind_pol]=aee*ell_template
        ind_pol+=1
    if inc_b :
        cl_out[ind_pol]=abb*ell_template
        ind_pol+=1
    if (inc_t and inc_e) :
        cl_out[ind_pol]=ate*ell_template


def cmb_setup(dictp) :
    dict_out={}
    for k in ['cl_tt_t','cl_ee_t','cl_bb_t','cl_te_t',
              'cl_tt_l','cl_ee_l','cl_bb_l','cl_te_l'] :
        if k in dictp.keys() :
            dict_out[k]=dictp[k].copy()
    return dict_out
    
def cmb_sed(nu):
    """ Function to compute CMB SED.

    Parameters
    ----------
    nu: float, or array_like(float)
        Frequency in GHz.
    """
    x = 0.0176086761 * nu
    ex = np.exp(x)
    sed = ex * (x / (ex - 1)) ** 2
    return sed

def cmb_cl(dict_cl,ell,r_tensor,a_lens,inc_t=False,inc_e=True,inc_b=True) :
    """ Function computing the CMB power spectrum

    Parameters
    ----------
    dict_cl:
        Dictionary containing the power spectrum templates
    ell: array_like
        Multipoles at which you want the power spectrum
    r_tensor: float
        Amplitude of tensor perturbations
    a_lens: float
        Lensing amplitude
    inc_t: bool
        Contains temperature?
    inc_e: bool
        Contains E-modes?
    inc_b: bool
        Contains B-modes?
    """
    npol=int(inc_t)+int(inc_e)+int(inc_b)+int(inc_t and inc_e)
    nell=len(ell)
    cl_out=np.zeros([npol,nell])
    ind_pol=0
    if inc_t :
        cl_out[ind_pol]=a_lens*dict_cl['cl_tt_l'][ell]+r_tensor*dict_cl['cl_tt_t'][ell]
        ind_pol+=1
    if inc_e :
        cl_out[ind_pol]=a_lens*dict_cl['cl_ee_l'][ell]+r_tensor*dict_cl['cl_ee_t'][ell]
        ind_pol+=1
    if inc_b :
        cl_out[ind_pol]=a_lens*dict_cl['cl_bb_l'][ell]+r_tensor*dict_cl['cl_bb_t'][ell]
        ind_pol+=1
    if (inc_t and inc_e) :
        cl_out[ind_pol]=a_lens*dict_cl['cl_te_l'][ell]+r_tensor*dict_cl['cl_te_t'][ell]

    return cl_out
        
def syncpl_setup(dictp) :
    return {'name':'syncpl'}    

def syncpl_sed(nu, nu_ref_s, beta_s):
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
    x = nu / nu_ref_s
    sed = x ** beta_s
    return sed

def syncpl_cl(dict_cl,ell,att_s,aee_s,abb_s,ate_s,
              alpha_s,ell_ref_s,inc_t=False,inc_e=True,inc_b=True) :
    """ Function computing the synchrotron power spectrum

    Parameters
    ----------
    dict_cl:
        Dictionary containing the power spectrum templates
    ell: array_like
        Multipoles at which you want the power spectrum
    att_s,aee_s,abb_s,ate_s: float
        Amplitude of the power spectrum (in TT,EE,BB and TE respectively)
    alpha_s: float
        Spectral tilt (D_l \propto (ell/ell_ref)^(alpha_s-2))
    ell_ref_s: float
        Pivot multipole
    inc_t: bool
        Contains temperature?
    inc_e: bool
        Contains E-modes?
    inc_b: bool
        Contains B-modes?
    """
    return cl_power_law(dict_cl,ell,att_s,aee_s,abb_s,ate_s,alpha_s,ell_ref_s,inc_t,inc_e,inc_b)

def sync_curvedpl_setup(dictp) :
    return {'name':'sync_curvedpl'}
    
def sync_curvedpl_sed(nu, nu_ref_s, beta_s, beta_c):
    """ Function to compute curved synchrotron power law SED.

    Parameters
    ----------
    nu: float, or array_like(float)
        Frequency in GHz.
    beta_s: float
        Power law index in RJ units.
    beta_c: float
        Power law index curvature.

    Returns
    -------
    array_like(float)
        Synchroton SED relative to reference frequency.
    """
    x = nu / nu_ref_s
    sed = x ** (beta_s + beta_c * np.log(nu / nu_ref_s))
    return sed

def sync_curvedpl_cl(dict_cl,ell,att_s,aee_s,abb_s,ate_s,
                     alpha_s,ell_ref_s,inc_t=False,inc_e=True,inc_b=True) :
    """ Function computing the synchrotron power spectrum

    Parameters
    ----------
    dict_cl:
        Dictionary containing the power spectrum templates
    ell: array_like
        Multipoles at which you want the power spectrum
    att_s,aee_s,abb_s,ate_s: float
        Amplitude of the power spectrum (in TT,EE,BB and TE respectively)
    alpha_s: float
        Spectral tilt (D_l \propto (ell/ell_ref)^(alpha_s-2))
    ell_ref_s: float
        Pivot multipole
    inc_t: bool
        Contains temperature?
    inc_e: bool
        Contains E-modes?
    inc_b: bool
        Contains B-modes?
    """
    return cl_power_law(dict_cl,ell,att_s,aee_s,abb_s,ate_s,alpha_s,ell_ref_s,inc_t,inc_e,inc_b)

def dustmbb_setup(dictp) :
    return {'name':'dustmbb'}
    
def dustmbb_sed(nu, nu_ref_d, beta_d, T_d):
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
    x_to = 0.0479924466 * nu / T_d
    x_from = 0.0479924466 * nu_ref_d / T_d
    sed = (nu / nu_ref_d) ** (1 + beta_d) * (np.exp(x_from) - 1) / (np.exp(x_to) - 1)
    return sed

def dustmbb_cl(dict_cl,ell,att_d,aee_d,abb_d,ate_d,
               alpha_d,ell_ref_d,inc_t=False,inc_e=True,inc_b=True) :
    """ Function computing the synchrotron power spectrum

    Parameters
    ----------
    dict_cl:
        Dictionary containing the power spectrum templates
    ell: array_like
        Multipoles at which you want the power spectrum
    att_d,aee_d,abb_d,ate_d: float
        Amplitude of the power spectrum (in TT,EE,BB and TE respectively)
    alpha_d: float
        Spectral tilt (D_l \propto (ell/ell_ref)^(alpha_d-2))
    ell_ref_d: float
        Pivot multipole
    inc_t: bool
        Contains temperature?
    inc_e: bool
        Contains E-modes?
    inc_b: bool
        Contains B-modes?
    """
    return cl_power_law(dict_cl,ell,att_d,aee_d,abb_d,ate_d,alpha_d,ell_ref_d,inc_t,inc_e,inc_b)
