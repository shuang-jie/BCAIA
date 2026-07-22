#' Posterior covariate-varying covariance matrix
#'
#' Reconstructs the model-implied covariance matrix
#' \eqn{\Sigma(x) = \Lambda(x)\Lambda(x)' + \sigma^2 I} at a covariate value
#' \code{x} for every saved posterior draw, where
#' \eqn{\lambda_{jk}(x) = q_{jk} f_k' x}.
#'
#' @param fit A fitted \code{"bcaia"} object from \code{\link{bcaia}}.
#' @param x A numeric covariance-covariate vector of length \code{Pcov}
#'   (matching the columns of \code{Xcov}, including the intercept).
#' @param summary One of \code{"median"} (default), \code{"mean"}, or
#'   \code{"none"}. With \code{"none"} the full \code{J x J x nsamp} array of
#'   posterior draws is returned.
#'
#' @return A \code{J x J} matrix (posterior summary) or, when
#'   \code{summary = "none"}, a \code{J x J x nsamp} array.
#' @seealso \code{\link{posterior_cor}}
#' @export
posterior_Sigma <- function(fit, x, summary = c("median", "mean", "none")) {
  stopifnot(inherits(fit, "bcaia"))
  summary <- match.arg(summary)
  x <- as.numeric(x)
  J <- fit$data$J; K <- fit$data$K
  if (length(x) != fit$data$Pcov)
    stop("x must have length Pcov = ", fit$data$Pcov, ".")
  nsamp <- fit$nsamp
  out <- array(NA_real_, dim = c(J, J, nsamp))
  for (t in seq_len(nsamp)) {
    fx <- as.numeric(fit$samples$F[, , t] %*% x)          # length K
    Lam <- fit$samples$Q[, , t] * matrix(fx, J, K, byrow = TRUE)
    out[, , t] <- tcrossprod(Lam) + diag(fit$samples$sig2[t], J)
  }
  switch(summary,
         none = out,
         mean = apply(out, c(1, 2), mean),
         median = apply(out, c(1, 2), stats::median))
}

#' Posterior covariate-varying correlation matrix
#'
#' Like \code{\link{posterior_Sigma}} but converts each posterior covariance
#' draw to a correlation matrix before summarising.
#'
#' @inheritParams posterior_Sigma
#' @return A \code{J x J} correlation matrix, or a \code{J x J x nsamp} array
#'   when \code{summary = "none"}.
#' @export
posterior_cor <- function(fit, x, summary = c("median", "mean", "none")) {
  summary <- match.arg(summary)
  Sig <- posterior_Sigma(fit, x, summary = "none")
  cors <- array(apply(Sig, 3, stats::cov2cor), dim = dim(Sig))
  switch(summary,
         none = cors,
         mean = apply(cors, c(1, 2), mean),
         median = apply(cors, c(1, 2), stats::median))
}

#' Posterior interval for a scalar quantity across covariate settings
#'
#' Convenience helper returning the posterior median and a credible interval of
#' the correlation between two features across a set of covariate vectors.
#'
#' @param fit A fitted \code{"bcaia"} object.
#' @param j,k Feature indices.
#' @param xlist A list of covariance-covariate vectors (each length
#'   \code{Pcov}).
#' @param level Credible level. Default \code{0.95}.
#' @return A data frame with columns \code{setting}, \code{median},
#'   \code{lower}, \code{upper}.
#' @export
posterior_cor_pair <- function(fit, j, k, xlist, level = 0.95) {
  a <- (1 - level) / 2
  rows <- lapply(seq_along(xlist), function(s) {
    cors <- posterior_cor(fit, xlist[[s]], summary = "none")
    v <- cors[j, k, ]
    data.frame(setting = names(xlist)[s] %||% s,
               median = stats::median(v),
               lower = stats::quantile(v, a),
               upper = stats::quantile(v, 1 - a))
  })
  do.call(rbind, rows)
}

## null-coalescing helper
`%||%` <- function(a, b) if (is.null(a)) b else a

#' @export
print.bcaia <- function(x, ...) {
  cat("BCAIA fit (", x$model, " model)\n", sep = "")
  cat(sprintf("  data: n = %d, J = %d, Pmean = %d, Pcov = %d, K = %d\n",
              x$data$n, x$data$J, x$data$Pmean, x$data$Pcov, x$data$K))
  if (!is.null(x$data$s)) cat(sprintf("  subjects: s = %d\n", x$data$s))
  cat(sprintf("  MCMC: %d iters, burn-in %d, thin %d -> %d saved samples\n",
              x$niter, x$burnin, x$thin, x$nsamp))
  cat(sprintf("  runtime: %.1f min\n", x$runtime[3] / 60))
  invisible(x)
}
