The `MapLike` object computes the likelihood given `SkyModel` and
`InstrumentModel` objects. The methods it contains will compute:
$$
F(\mathbf b, \hat n) \\
\bar N_T^{-1}=F^TN^{-1}F \\
\bar{\mathbf T} = N_T F^T N^{-1}\mathbf d \\
\log[p(\mathbf b | \mathbf d)]
$$
