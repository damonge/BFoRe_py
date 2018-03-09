Issues:

- Update Component and SkyModel to have different SEDs for temperature and
polarization.
- Have a look at which parts of the likelihood need to be computed each time,
and which can be calculated once on initialization of MapLike.
  - For example d_map * n_ivar_map in calculation of amp_covar can be calculated
    once at the beginning of each sampling.
- Add a method to MapLike that can accept masks, and let split_data know about
it for default runs.
- Add method for maringal spectral likelihood to return best-fit amplitdue and
spectral parameter maps.
- Add method to calculate chi2 maps from amplitude maps for best-fit spec
params.
- Implement instrument model.


Plan for which parameters to vary:

  - Tell MapLike which parameters are variable.
  - MapLike can inspect SkyModel to find which parameters are required.
  - List of parameters in marginal_spectral_likelihood() has same ordering as
  list of variable parameters.
