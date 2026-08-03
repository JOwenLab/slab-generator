# Table T3 - Surface energies and their hydrogen chemical-potential slopes

| facet | gamma(H-rich) | scatter | dgamma/dmu | coverage | gamma=0 at | dehyd. ceiling | mu_C drift | gamma bias | level |
| ----- | ------------: | ------: | ---------: | -------: | ---------: | -------------: | ---------: | ---------: | ----- |
| (100) |       +0.0501 |  0.0241 |     2.5100 |   0.1567 |    -0.0200 |   NOT COMPUTED |    -0.0524 |    +0.0103 | L1    |
| (110) |       -0.8265 |  0.0003 |     3.5497 |   0.2216 |    +0.2328 |   NOT COMPUTED |    +0.0017 |    -0.0005 | L2    |
| (111) |       -0.9761 |  0.0006 |     2.8983 |   0.1809 |    +0.3368 |   NOT COMPUTED |    +0.0032 |    -0.0007 | L2    |

**Table T3.** Surface energy of the H-terminated facets and its hydrogen chemical-potential dependence: gamma(Delta mu_H) = gamma(H-rich) + (dgamma/dmu) Delta mu_H, with Delta mu_H measured BELOW the H-rich limit. The slope is the H coverage, so the ordering of the three facets changes with mu_H and the equilibrium habit changes with it -- that dependence is what F2 and F3 are built on. Negative gamma at the H-rich limit is not an error: it means H termination is favourable against the H2 reservoir there, but a Wulff construction is undefined while any gamma < 0, which is why the shape at that end is not computed. The mu_C drift column is the fitted carbon chemical potential's departure from the bulk reference and the gamma bias is what that drift costs; on (100) it is 15x larger than on either other facet, which is why that row is L1 while the others are L2. The dehydrogenation ceiling -- the Delta mu_H beyond which a bare facet is more stable -- is NOT COMPUTED: it requires a bare-facet ladder that is not in this repository, so the hydrogen-poor side of the window is unbounded and any statement made there is an extrapolation with no stop.


- Units: gamma and scatter J/m^2; dgamma/dmu J m^-2 eV^-1; coverage n_H/2A per A^2; crossings and ceiling eV below the H-rich limit; mu_C drift mRy; gamma bias J/m^2.
- Scatter is the spread of per-slab gamma across the fitted thickness ladder, i.e. reproducibility within the ladder, not accuracy.
