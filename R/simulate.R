#' Simulate data from the covariate-varying interaction model (Simulation 1)
#'
#' Generates a dataset mirroring Simulation 1 of the paper: a categorical
#' design with two factors (a binary and a ternary covariate) giving six
#' experimental conditions, a rank-\code{K_true} covariate-varying covariance,
#' and mean-abundance regression coefficients. Useful for quick demonstrations
#' and for reproducing the simulation study.
#'
#' @param n Number of samples (must be a multiple of 6). Default \code{30}.
#' @param J Number of features. Default \code{15}.
#' @param K_true True number of covariate-varying factors. Default \code{2}.
#' @param sigma True idiosyncratic standard deviation. Default \code{0.5}.
#' @param seed Optional integer seed.
#'
#' @return A list with the count matrix \code{Y}, the design matrices
#'   \code{Xmean} and \code{Xcov}, and a \code{truth} sublist containing
#'   \code{Sigma} (a \code{J x J x n} array), \code{Q}, \code{F}, \code{beta},
#'   \code{ri}, \code{alpha} and \code{mu}.
#' @export
#'
#' @examples
#' sim <- simulate_bcaia(seed = 6)
#' dim(sim$Y)
simulate_bcaia <- function(n = 30, J = 15, K_true = 2, sigma = 0.5, seed = NULL) {
  if (n %% 6 != 0) stop("n must be a multiple of 6 (six experimental conditions).")
  if (!is.null(seed)) set.seed(seed)
  per <- n / 6

  ## Mean design: binary main effect + ternary main effect (no intercept)
  Xmean <- cbind(rep(c(1, 0), each = n / 2),
                 rep(c(0, 1), each = n / 2),
                 rep(rep(c(1, 0, 0), each = per), 2),
                 rep(rep(c(0, 1, 0), each = per), 2),
                 rep(rep(c(0, 0, 1), each = per), 2))
  Pmean <- ncol(Xmean)

  ## Covariance design (intercept + two indicators), six conditions
  conditions <- rbind(c(1, 0, 0, 0), c(1, 0, 1, 0), c(1, 0, 0, 1),
                      c(1, 1, 0, 0), c(1, 1, 1, 0), c(1, 1, 0, 1))
  Xcov <- conditions[rep(seq_len(6), each = per), ]
  Pcov <- ncol(Xcov)

  ## True F with the "factor drops out at certain x" constraints
  F.true <- matrix(stats::runif(K_true * Pcov, -1, 1), K_true, Pcov, byrow = TRUE)
  F.true[2, 1] <- stats::runif(1, 1.1, 1.2)
  F.true[2, 2] <- -F.true[2, 1]
  F.true[1, 3] <- -F.true[1, 1]

  ## True Q: zero w.p. 1/2, else standard normal shifted off zero by 1
  Q.true <- matrix(0, J, K_true)
  for (j in seq_len(J)) for (k in seq_len(K_true)) {
    if (stats::runif(1) < 0.5) {
      t <- extraDistr::rtnorm(1, 0, 1, -1, 1)
      Q.true[j, k] <- t + (t > 0) - (t < 0)
    }
  }

  FX <- tcrossprod(Xcov, F.true)
  Sigma <- array(NA_real_, dim = c(J, J, n))
  for (i in seq_len(n)) {
    Lam <- Q.true * matrix(FX[i, ], J, K_true, byrow = TRUE)
    Sigma[, , i] <- tcrossprod(Lam) + diag(sigma^2, J)
  }

  ## Mean abundance
  ri <- stats::runif(n, 0, 2)
  comp <- stats::rbinom(J, 1, 0.7)
  alpha <- ifelse(comp == 1, stats::rnorm(J, 5, 0.5), stats::rnorm(J, -1, 1))
  beta <- matrix(0, J, Pmean)
  for (j in seq_len(J)) for (p in seq_len(Pmean)) {
    if (stats::runif(1) >= 0.5) {
      cv <- stats::rnorm(1)
      beta[j, p] <- cv + (cv > 0) - (cv < 0)
    }
  }
  mu <- matrix(ri, n, J) + matrix(alpha, n, J, byrow = TRUE) + tcrossprod(Xmean, beta)

  Y <- matrix(0, n, J)
  for (i in seq_len(n)) {
    ystar <- mu[i, ] + as.numeric(.rmvn_chol(Sigma[, , i]))
    Y[i, ] <- floor(exp(ystar))
  }

  list(Y = Y, Xmean = Xmean, Xcov = Xcov,
       truth = list(Sigma = Sigma, Q = Q.true, F = F.true, beta = beta,
                    ri = ri, alpha = alpha, mu = mu))
}

## Internal: one draw from N(0, S) via Cholesky (avoids extra dependencies).
.rmvn_chol <- function(S) {
  L <- chol(S + diag(1e-10, nrow(S)))
  as.numeric(crossprod(L, stats::rnorm(nrow(S))))
}
