#' Bayesian covariate-varying interaction analysis for multivariate counts
#'
#' Fits the Bayesian covariate-varying factor model of Zhang, Patnode and Lee
#' by MCMC. The model jointly estimates a covariate-dependent mean abundance and
#' a covariate-dependent covariance (interaction) structure for high-dimensional
#' multivariate count data, using a rounded multivariate log-normal kernel, a
#' sparse covariate-dependent factor loading matrix with a Dirichlet-Horseshoe
#' prior, and mean-constrained Dirichlet process mixtures for the size factor
#' and baseline abundance.
#'
#' Two model variants are available and selected automatically through the
#' \code{subject} argument:
#' \itemize{
#'   \item \strong{Simple model} (\code{subject = NULL}): one baseline abundance
#'     \eqn{\alpha_j} per feature. This matches Simulation 1 and the mice data
#'     analysis in the paper.
#'   \item \strong{Subject-indexed model} (\code{subject} supplied): a
#'     subject-specific baseline abundance \eqn{\alpha_{s,j}} to accommodate
#'     repeated samples and large inter-subject variability. This matches
#'     Simulations 2 and 3.
#' }
#'
#' @param Y An \code{n x J} matrix of non-negative integer counts (samples in
#'   rows, features/OTUs in columns).
#' @param Xmean An \code{n x Pmean} design matrix for the mean regression
#'   \emph{excluding} the intercept (the intercept is absorbed into the
#'   size factor / baseline abundance).
#' @param Xcov An \code{n x Pcov} design matrix for the covariance regression,
#'   \emph{including} an intercept column (typically the first column of ones).
#' @param subject Optional length-\code{n} vector (factor, integer or character)
#'   giving the subject each sample belongs to. When supplied, the
#'   subject-indexed model is fitted; when \code{NULL}, the simple model is
#'   fitted.
#' @param K Number of latent factors (the working dimension of the loading
#'   matrix). Choose a value large enough to capture the covariance; redundant
#'   factors are shrunk out by the prior. See \code{\link{choose_K}}.
#' @param niter Total number of MCMC iterations. Default \code{160000}.
#' @param burnin Number of burn-in iterations discarded before saving. Default
#'   \code{niter/2}.
#' @param thin Thinning interval for saved samples. Default \code{10}.
#' @param seed Optional integer seed for reproducible initialisation.
#' @param control A \code{\link{bcaia_control}} list of hyperparameters and
#'   tuning constants.
#' @param verbose Logical; print progress. Default \code{TRUE}.
#'
#' @return An object of class \code{"bcaia"}: a list with element \code{samples}
#'   (posterior draws of \code{F}, \code{Q}, \code{tau}, \code{beta}, \code{ri},
#'   \code{alpha} / \code{alphasij}, \code{sig2}), together with \code{runtime},
#'   \code{data} (dimensions and the design matrices), \code{control}, and the
#'   \code{model} variant used. Use \code{\link{posterior_Sigma}} and
#'   \code{\link{posterior_cor}} to summarise covariate-varying covariance and
#'   correlation.
#'
#' @references Zhang, S., Patnode, M. L. and Lee, J. Bayesian Covariate-Varying
#'   Interaction Analysis for Multivariate Count Data: Application to Microbiome
#'   Studies.
#'
#' @seealso \code{\link{bcaia_control}}, \code{\link{simulate_bcaia}},
#'   \code{\link{posterior_Sigma}}, \code{\link{choose_K}}
#' @export
bcaia <- function(Y, Xmean, Xcov,
                  subject = NULL,
                  K = 8,
                  niter = 160000,
                  burnin = NULL,
                  thin = 10,
                  seed = NULL,
                  control = bcaia_control(),
                  verbose = TRUE) {

  Y <- as.matrix(Y)
  Xmean <- as.matrix(Xmean)
  Xcov <- as.matrix(Xcov)
  storage.mode(Y) <- "double"

  n <- nrow(Y); J <- ncol(Y)
  if (nrow(Xmean) != n || nrow(Xcov) != n)
    stop("Xmean and Xcov must have the same number of rows as Y.")
  if (any(Y < 0) || any(Y != floor(Y)))
    stop("Y must contain non-negative integer counts.")
  if (is.null(burnin)) burnin <- niter %/% 2
  if (is.null(control$b_tau)) control$b_tau <- 1 / J

  if (is.null(subject)) {
    .bcaia_simple(Y, Xmean, Xcov, K, niter, burnin, thin, seed, control, verbose)
  } else {
    .bcaia_subject(Y, Xmean, Xcov, subject, K, niter, burnin, thin, seed,
                   control, verbose)
  }
}


## ---------------------------------------------------------------------------
## Simple model (alpha_j): port of the Simulation 1 / mice-data sampler.
## ---------------------------------------------------------------------------
.bcaia_simple <- function(Y, Xmean, Xcov, K, niter, burnin, thin, seed,
                          control, verbose) {
  n <- nrow(Y); J <- ncol(Y)
  Pmean <- ncol(Xmean); Pcov <- ncol(Xcov)
  cc <- control
  if (!is.null(seed)) set.seed(seed)

  ## --- data-driven initial values ---
  hat.ri <- rowSums(log(Y + 0.01) / J)
  hat.alpha <- colMeans(log(Y + 0.01) - matrix(hat.ri, n, J))
  hat.alphaij <- log(Y + 0.01) - matrix(hat.ri, n, J)

  nu.r <- mean(hat.ri); a.xi <- nu.r
  ri <- hat.ri
  Si1 <- sample(seq_len(cc$Lr), n, replace = TRUE)
  Si2 <- sample(0:1, n, replace = TRUE)
  w.l.r <- stats::rbeta(cc$Lr, cc$a_w, cc$b_w)
  V.r <- stats::rbeta(cc$Lr - 1, 1, cc$a_psi_r)
  psi.r <- .recover_psi(V.r)
  xi <- rep(0, cc$Lr)

  nu.alpha <- 0
  u2.alpha <- mean(Rfast::colVars(hat.alphaij))
  a.xi.alpha <- nu.alpha
  Sj1 <- sample(seq_len(cc$L_alpha), J, replace = TRUE)
  Sj2 <- rep(1, J)
  xi.alpha <- stats::rnorm(cc$L_alpha, nu.alpha, sqrt(u2.alpha))
  w.alpha <- stats::rbeta(cc$L_alpha, cc$a_w_alpha, cc$b_w_alpha)
  V.alpha <- stats::rbeta(cc$L_alpha - 1, 1, cc$a_psi_alpha)
  psi.alpha <- .recover_psi(V.alpha)

  ## --- parameter initialisation ---
  eta <- matrix(stats::rnorm(n * K), n, K)
  F.m <- matrix(stats::rnorm(K * Pcov), K, Pcov)
  y.star <- log(Y + 0.01)
  Q <- matrix(stats::rnorm(J * K), J, K)
  Lambdai.array <- update_Lami(Q, F.m, Xcov, J, K, n)
  ZZ <- matrix(1 / stats::rgamma(J * K, 1 / 2, 1), J, K)
  zzeta <- matrix(1 / stats::rgamma(J * K, 1 / 2, 1 / ZZ), J, K)
  tau <- rep(1, K)
  phi <- til.phi <- matrix(0, J, K)
  for (ki in seq_len(K)) {
    g <- stats::rgamma(J, cc$a_phi, 1)
    phi[, ki] <- g / sum(g); til.phi[, ki] <- g
  }
  ri.muij <- matrix(ri, n, J)
  alphaj <- hat.alpha
  alphaij <- matrix(alphaj, n, J, byrow = TRUE)
  betajp <- matrix(stats::rnorm(J * Pmean), J, Pmean)
  sig2 <- 1 / stats::rgamma(1, cc$a_sig, cc$b_sig)

  sig.pro.phi <- matrix(1, J, K)
  nacc.phi <- true.nacc.phi <- acc.phi <- matrix(0, J, K)
  log.Y <- log(Y); log.Y1 <- log(Y + 1)
  RSS <- t(vapply(seq_len(n), function(x)
    y.star[x, ] - tcrossprod(eta[x, ], Lambdai.array[, , x]), numeric(J))) -
    ri.muij - alphaij - tcrossprod(Xmean, betajp)

  sig_pro <- matrix(1, K, Pcov); mu_adap <- matrix(0, K, Pcov)
  nacpt.Fkp <- true.nacpt.Fkp <- matrix(0, K, Pcov)
  loglambda_adap <- matrix(log(2.38^2 / Pcov), K, Pcov)
  sig.pro.wla <- rep(1, cc$L_alpha); mu_adap_w_a <- rep(0, cc$L_alpha)
  nacc.w.a <- true.nacc.w.a <- rep(0, cc$L_alpha)
  loglbd_wa <- rep(log(2.38^2), cc$L_alpha)
  sig.pro.wlr <- rep(1, cc$Lr); mu_adap_w_r <- rep(0, cc$Lr)
  nacc.w.r <- true.nacc.w.r <- rep(0, cc$Lr)
  loglbd_wr <- rep(log(2.38^2), cc$Lr)

  nsamp <- floor((niter - burnin) / thin); count.st <- 0
  F.st <- array(NA_real_, dim = c(K, Pcov, nsamp))
  Q.st <- array(NA_real_, dim = c(J, K, nsamp))
  tau.st <- matrix(0, K, nsamp)
  beta.st <- array(NA_real_, dim = c(J, Pmean, nsamp))
  ri.st <- matrix(0, nsamp, n); alphaj.st <- matrix(0, nsamp, J)
  sig2.st <- numeric(nsamp)

  if (verbose) cat(sprintf("BCAIA (simple model): %d iters, burn-in %d, thin %d\n",
                           niter, burnin, thin))
  t0 <- proc.time()

  for (ni in seq_len(niter)) {
    RSS <- y.star - RSS
    y.star <- update_ystar(log.Y, log.Y1, n, J, RSS, sig2)
    RSS <- y.star - RSS

    sig2 <- update_sig2(n, J, RSS, cc$a_sig, cc$b_sig)

    delta <- min(0.01, 1 / sqrt(ni))
    RSS <- RSS + t(vapply(seq_len(n), function(x)
      tcrossprod(eta[x, ], Lambdai.array[, , x]), numeric(J)))
    eeta <- tcrossprod(Xcov, F.m) * eta
    Q <- update_Q(J, K, zzeta, tau, phi, eeta, RSS, sig2)
    Lambdai.array <- update_Lami(Q, F.m, Xcov, J, K, n)

    zz <- update_ZZ_ZZeta(J, K, Q, phi, tau, ZZ, zzeta)
    zzeta <- zz[, , 1]; ZZ <- zz[, , 2]

    ph <- update_phi(K, J, til.phi, sig.pro.phi, cc$a_phi, Q, zzeta, tau,
                     acc.phi, phi, nacc.phi, true.nacc.phi, ni, cc$acc_tar, delta)
    til.phi <- ph[, , 1]; phi <- ph[, , 2]
    nacc.phi <- ph[, , 3]; true.nacc.phi <- ph[, , 4]; sig.pro.phi <- ph[, , 5]

    tau <- update_tau_GIG(K, cc$a_tau, cc$b_tau, J, Q, phi, zzeta)

    gamma_adap <- min(0.5, 1 / (ni^(2 / 3)))
    ff <- update_F_kp_MCMC_signswitching(n, J, K, Pcov, F.m, Q, Xcov, sig_pro,
                                         sig2, RSS, nacpt.Fkp, true.nacpt.Fkp,
                                         cc$acc_tar, ni, gamma_adap,
                                         loglambda_adap, mu_adap)
    F.m <- abind::adrop(ff[, , 1, drop = FALSE], 3)
    mu_adap <- abind::adrop(ff[, , 2, drop = FALSE], 3)
    nacpt.Fkp <- abind::adrop(ff[, , 3, drop = FALSE], 3)
    true.nacpt.Fkp <- abind::adrop(ff[, , 4, drop = FALSE], 3)
    sig_pro <- abind::adrop(ff[, , 5, drop = FALSE], 3)
    loglambda_adap <- abind::adrop(ff[, , 6, drop = FALSE], 3)
    Lambdai.array <- update_Lami(Q, F.m, Xcov, J, K, n)

    eta <- update_Delta(K, Lambdai.array, sig2, n, RSS)
    RSS <- RSS - t(vapply(seq_len(n), function(x)
      tcrossprod(eta[x, ], Lambdai.array[, , x]), numeric(J)))

    psi.r <- c(update_psi_a(cc$Lr, Si1 - 1, Si2, cc$a_psi_r))
    RSS <- RSS + ri.muij
    wr <- update_w_r_new(w.l.r, cc$Lr, Si1 - 1, Si2, cc$a_w, cc$b_w,
                         sig.pro.wlr, ri, loglbd_wr, sig2, xi, nu.r,
                         nacc.w.r, true.nacc.w.r, gamma_adap, 0.44, mu_adap_w_r, n)
    w.l.r <- wr[, 1]; mu_adap_w_r <- wr[, 2]; sig.pro.wlr <- wr[, 3]
    s12 <- update_Si12(cc$Lr, n, ri, xi, cc$ur2, nu.r, w.l.r, psi.r)
    Si1 <- s12[, 1] + 1; Si2 <- s12[, 2]
    xi <- update_xi(cc$Lr, ri, Si1 - 1, Si2, a.xi, 1, w.l.r, nu.r, cc$ur2)
    ri <- update_ri(n, cc$ur2, J, sig2, RSS, Si1 - 1, Si2, nu.r, xi, w.l.r)
    ri.muij <- matrix(ri, n, J)
    RSS <- RSS - ri.muij

    psi.alpha <- c(update_psi_a(cc$L_alpha, Sj1 - 1, Sj2, cc$a_psi_alpha))
    RSS <- RSS + alphaij
    wa <- update_w_a(w.alpha, cc$L_alpha, Sj1 - 1, Sj2, cc$a_w_alpha, cc$b_w_alpha,
                     sig.pro.wla, RSS, loglbd_wa, sig2, xi.alpha, nu.alpha,
                     nacc.w.a, true.nacc.w.a, gamma_adap, 0.44, mu_adap_w_a)
    w.alpha <- wa[, 1]; mu_adap_w_a <- wa[, 2]; sig.pro.wla <- wa[, 3]
    sj <- update_Sj12(RSS, J, cc$L_alpha, xi.alpha, nu.alpha, w.alpha, psi.alpha, sig2)
    Sj1 <- sj[, 1] + 1; Sj2 <- sj[, 2]
    xi.alpha <- update_xi_alpha(cc$L_alpha, n, J, RSS, Sj1 - 1, Sj2, u2.alpha,
                                a.xi.alpha, w.alpha, sig2, nu.alpha)
    alphaj <- vapply(seq_len(J), function(j)
      Sj2[j] * xi.alpha[Sj1[j]] +
        (1 - Sj2[j]) * (nu.alpha - w.alpha[Sj1[j]] * xi.alpha[Sj1[j]]) /
        (1 - w.alpha[Sj1[j]]), numeric(1))
    alphaij <- matrix(alphaj, n, J, byrow = TRUE)
    RSS <- RSS - alphaij

    RSS <- RSS + tcrossprod(Xmean, betajp)
    for (ji in seq_len(J)) {
      U.l <- chol(diag(rep(1 / cc$u2_beta, Pmean)) + crossprod(Xmean) / sig2 +
                    diag(rep(1e-10, Pmean)))
      a.l <- crossprod(Xmean, RSS[, ji]) / sig2
      betajp[ji, ] <- backsolve(U.l, backsolve(U.l, a.l, transpose = TRUE) +
                                  stats::rnorm(Pmean))
    }
    RSS <- RSS - tcrossprod(Xmean, betajp)

    if (ni > burnin && ni %% thin == 0) {
      count.st <- count.st + 1
      Q.st[, , count.st] <- Q; F.st[, , count.st] <- F.m
      tau.st[, count.st] <- tau; beta.st[, , count.st] <- betajp
      ri.st[count.st, ] <- ri; alphaj.st[count.st, ] <- alphaj
      sig2.st[count.st] <- sig2
    }
    if (verbose && ni %% 1000 == 0) cat("  iter", ni, "\r")
  }
  run.time <- proc.time() - t0
  if (verbose) cat(sprintf("\nDone in %.1f min; saved %d samples.\n",
                           run.time[3] / 60, count.st))

  structure(list(
    model = "simple",
    samples = list(F = F.st, Q = Q.st, tau = tau.st, beta = beta.st,
                   ri = ri.st, alpha = alphaj.st, sig2 = sig2.st),
    runtime = run.time,
    data = list(n = n, J = J, Pmean = Pmean, Pcov = Pcov, K = K,
                Xmean = Xmean, Xcov = Xcov),
    control = cc, niter = niter, burnin = burnin, thin = thin, nsamp = count.st
  ), class = "bcaia")
}
