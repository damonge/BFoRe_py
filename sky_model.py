import numpy as np
from components import ComponentBase,ComponentCMB,ComponentSyncPL,ComponentDustMBB

class SkyModel(object) :
    """
    Sky model. Basically defined by a 
    set of components and some extra parameters (e.g. polarization/temp)
    """
    comps=[]
    
    def __init__(self,is_polarized=False,include_cmb=True,
                 include_sync=True,include_dust=True,extra_components=None) :
        """
        Initializes a sky model
        is_polarized (bool): determines whether maps are I or (Q,U)
        include_cmb (bool): is CMB one of the components?
        include_sync (bool): is synchrotron one of the components?
        include_dust (bool): is dust one of the components?
        extra_components (array_like) : array of extra ComponentXYZ objects
        """

        #Basically stack components together
        #...
        if include_cmb :
            comps.append(ComponentCMB())
        if include_sync :
            comps.append(ComponentSyncPL())
        if include_dust :
            comps.append(ComponentDustMBB())
        if extra_components is not None :
            for c in extra_components :
                comps.append(c)
        
        self.ncomp=len(comps)
        
    def fnu(self,nu,params) :
        """
        Return matrix of SEDs
        nu (array_like) : frequencies 
        params : parameters for all the SEDs
        QUESTION : we should think about what the best way of passing these parameters is.
                   One option would be to pass a dictionary, such that params['sync']
                   be the parameters for the synchrotron model.
        """
        fnu=np.zeros([self.ncomp,len(nu)])
        for i,c in enumerate(self.comp) :
            fnu[i,:]=c.fnu(nu,params)

        return fnu
