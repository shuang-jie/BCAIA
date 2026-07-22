# BCAIA

<!-- badges: start -->
<!-- badges: end -->

**B**ayesian **C**ovariate-varying **I**nteraction **A**nalysis for multivariate
count data.

`BCAIA` implements the Bayesian covariate-varying factor model of Zhang,
Patnode and Lee for high-dimensional multivariate count data, with a focus on
microbiome studies. The model **jointly** estimates

* how the **mean abundance** of each feature changes with covariates, and
* how the **interaction (covariance) structure** among features changes with
  covariates,

directly on discrete, over-dispersed, zero-inflated counts — without ad-hoc
log/clr transformations.

The covariance is modelled through a sparse covariate-dependent factor loading
matrix,

<p align="center">
&Sigma;(x) = &Lambda;(x)&Lambda;(x)' + &sigma;<sup>2</sup> I,
&nbsp;&nbsp; &lambda;<sub>jk</sub>(x) = q<sub>jk</sub> f<sub>k</sub>'x,
</p>

with a **Dirichlet–Horseshoe** prior on the loadings for joint sparsity, a
**rounded multivariate log-normal kernel** for the counts, and
**mean-constrained Dirichlet process mixtures** for the size factor and baseline
abundance.

## Installation

```r
# install.packages("devtools")
devtools::install_github("shuang-jie/BCAIA")
```

The package compiles C++ (via Rcpp / RcppArmadillo), so you need a working
toolchain (Rtools on Windows, Xcode command line tools on macOS).

## Quick start

```r
library(BCAIA)

## simulate a small covariate-varying dataset (Simulation 1 design)
sim <- simulate_bcaia(n = 30, J = 15, seed = 6)

## choose the working number of factors from a clr-PCA scree rule
K <- choose_K(sim$Y)

## fit the model
fit <- bcaia(Y = sim$Y, Xmean = sim$Xmean, Xcov = sim$Xcov,
             K = 8, niter = 160000, seed = 1)

## posterior covariance / correlation at a covariate setting
Sigma_hat <- posterior_Sigma(fit, x = sim$Xcov[1, ])
rho_hat   <- posterior_cor(fit,   x = sim$Xcov[1, ])
```

### Repeated samples / inter-subject variability

Supply a `subject` vector to fit the subject-indexed variant (a
subject-specific baseline abundance &alpha;<sub>s,j</sub>), used in
Simulations 2 and 3:

```r
fit2 <- bcaia(Y, Xmean, Xcov, subject = subject_id, K = 7, niter = 160000)
```

### Real data

The mice gut microbiome data of Patnode et al. (2019) analysed in the paper are
bundled:

```r
data(mice)
fit_mice <- bcaia(mice$Y, mice$Xmean, mice$Xcov, K = 8, niter = 160000)
```

## Reproducing the paper

The `vignettes/` reproduce the paper's analyses:

| Vignette | Paper section |
|---|---|
| `simulation1` | §4.1 Simulation 1 (categorical design) |
| `simulation2` | §4.2 Simulation 2 (repeated samples) |
| `simulation3` | §4.3 Simulation 3 (arbitrary covariance) |
| `mice`        | §5 Mice gut microbiome data |

Each vignette runs a short chain for illustration; the full paper settings
(`niter = 160000`) are noted inline and are best run on a server, optionally
over multiple seeds/replicates.

## Reference

Zhang, S., Patnode, M. L. and Lee, J. *Bayesian Covariate-Varying Interaction
Analysis for Multivariate Count Data: Application to Microbiome Studies.*

## License

MIT © Shuangjie Zhang
