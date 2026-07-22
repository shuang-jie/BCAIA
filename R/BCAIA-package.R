#' BCAIA: Bayesian Covariate-Varying Interaction Analysis for Count Data
#'
#' The package fits the Bayesian covariate-varying factor model of Zhang,
#' Patnode and Lee for high-dimensional multivariate count data, jointly
#' estimating covariate-dependent mean abundance and covariate-dependent
#' interaction (covariance) structure. The main entry point is
#' \code{\link{bcaia}}; posterior summaries are obtained with
#' \code{\link{posterior_Sigma}} and \code{\link{posterior_cor}}.
#'
#' @keywords internal
#' @useDynLib BCAIA, .registration = TRUE
#' @importFrom Rcpp sourceCpp
## GIGrvg::rgig is called from the C++ engine (rgigRcpp); import it so the
## dependency is recognised by R CMD check.
#' @importFrom GIGrvg rgig
"_PACKAGE"
