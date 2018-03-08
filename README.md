Issues:

- More robust way of passing which parameters are going to be variable to the
MapLike, and SkyModel. How to combine this with argument passing to the
MapLike.marginal_spectral_likelihood() method, which requires the variable
args to be passed as a list.
- Update Component and SkyModel to have different SEDs for temperature and
polarization.
- Have a look at which parts of the likelihood need to be computed each time,
and which can be calculated once on initialization of MapLike.
  - For example d_map * n_ivar_map in calculation of amp_covar can be calculated
    once at the beginning of each sampling. 
- Add a method to MapLike that can accept masks, and let split_data know about
it for default runs.
- Do we want to implement an alternative method to sample amplitudes jointly?
- Add method for maringal spectral likelihood to return best-fit amplitdue and
spectral parameter maps.
- Add method to calculate chi2 maps from amplitude maps for best-fit spec
params.
