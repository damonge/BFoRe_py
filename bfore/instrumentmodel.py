import numpy as np

class InstrumentModel(object) :
    """
    Instrument model.
    Currently this is mostly a glorified 2D array containing bandpasses for each frequency channel.
    """

    def __init__(self,bandpasses) :
        """
        Initializes an instrument model
        bandpasses (array_like): an array of dictionaries for each frequency channel. Each dictionary should contain 2 fields: 'nu', and 'bps'. 'bps' should be an array with N values containing the spectral transmission in each of N adjacent frquency bins. 'nu' should be an array with N+1 values containing the edges of the frequency bins (in GHz). Note that we assume that the bandpasses are normalized for a constant spectrum in units of antenna temperature K_RJ.
        """
        self.n_channels=len(bandpasses)
        self.nu_arrs=np.array([0.5*(b['nu'][1:]+b['nu'][:-1]) for b in bandpasses])
        self.bps_arrs=np.array([b['bps']*(b['nu'][1:]-b['nu'][:-1]) for b in bandpasses])

        #Normalize bandpasses
        for i in np.arange(self.n_channels) :
            self.bps_arrs[i]/=np.sum(self.bps_arrs[i])

    def convolve_sed(self,sed,args=None,instpar=None) :
        """
        Convolves a given SED with each of the bandpasses, returning a vector, with one element per bandpass response.
        sed (function) : a function taking two arguments: an array of frequencies (in GHz) and an array of parameters that define the SED.
        args (array_like) : set of parameters to pass to 'sed'
        instpar (array_like) : array of additional parameters associated with this instrument (e.g. bandpass or gain shifts). None are implemented yet.
        """
        #Returns Ncomp x Nfreq array
        return np.transpose(np.array([np.sum(b*sed(n,args),axis=1) for n,b in zip(self.nu_arrs,self.bps_arrs)])) #TODO: check/optimize
