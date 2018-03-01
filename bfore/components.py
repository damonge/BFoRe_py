import numpy as np

class ComponentBase(object) :
    """
    Empty component class   
    No parameters
    """
    def __init__(self,comp_name='base') :
        self.comp_name=comp_name
        pass

    def fnu(nu,p) :
        """
        Return this component's SED
        nu (array_like) : Frequency (in GHz)
        p (array_like) : array of parameters defining the SED
        Question: should we assume uK_RJ by default and let the initializer take care of units?
        """
        return np.ones_like(nu)

class ComponentCMB(ComponentBase) :
    """
    CMB component
    No parameters
    """

    def __init__(self,comp_name='cmb') :
        self.comp_name=comp_name

    def fnu(nu,p) :
        x=0.0176086761*nu
        ex=np.exp(x)
        return ex*(x/(ex-1))**2

class ComponentSyncPL(ComponentBase) :
    """
    Synchrotron power-law component
    p[0]=nu_0
    p[1]=beta_s
    """

    def __init__(self,comp_name='sync_pl') :
        self.comp_name=comp_name

    def fnu(nu,p) :
        pars=p[self.comp_name]
        x=nu/pars[0]
        return x**pars[1]

class ComponentDustMBB(ComponentBase) :
    """
    Dust modified-BB component
    p[0]=nu_0
    p[1]=beta_d
    p[2]=temp_d [K]
    """

    def __init__(self,comp_name='dust_mbb') :
        self.comp_name=comp_name

    def fnu(nu,p) :
        pars=p[self.comp_name]
        x_to=0.0479924466*nu/pars[2]
        x_from=0.0479924466*pars[0]/pars[2]
        return (nu/pars[0])**(1+pars[1])*(np.exp(x_from)-1)/(np.exp(x_to)-1)
