from __future__ import print_function
import numpy as np

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

