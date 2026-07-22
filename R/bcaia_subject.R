## ---------------------------------------------------------------------------
## Subject-indexed model (alpha_{s,j}): port of the Simulation 2 / 3 sampler.
##
## A subject-specific baseline abundance is shared by all samples from a
## subject. This implementation covers the single feature-group case (m = 1),
## which is what the paper's Simulations 2 and 3 use; the covariate-varying
## covariance (including the baseline factor) is captured through the intercept
## column of Xcov. Called from bcaia() when `subject` is supplied.
## ---------------------------------------------------------------------------
.bcaia_subject <- function(Y, Xmean, Xcov, subject, K, niter, burnin, thin,
                           seed, control, verbose) {
  n <- nrow(Y); J <- ncol(Y)
  Pmean <- ncol(Xmean); Pcov <- ncol(Xcov)
  cc <- control
  if (!is.null(seed)) set.seed(seed)

  ## subject membership 1..s (single feature group m = 1)
  S <- as.integer(factor(subject))
  s <- length(unique(S))
  m <- 1L
  Jvec <- J                 # features per group (rowvec of length m)
  Mgrp <- rep(1L, J)        # feature -> group map (1-indexed)
  J_bound <- c(0, J - 1)    # 0-indexed inclusive column boundaries per group

  ## --- data-driven initial values ---
  hat.ri <- matrix(rowSums(log(Y + 0.01) / J), 1, n)
  hat.alpha <- colMeans(log(Y + 0.01) - matrix(hat.ri, n, J))
  hat.alphasij <- matrix(0, s, J)
  for (i in seq_len(s)) for (j in seq_len(J))
    hat.alphasij[i, j] <- mean(log(Y[S == i, j] + 0.01) - hat.ri[1, S == i])

  nu.r <- rowMeans(hat.ri); a.xi <- nu.r
  ri <- hat.ri
  Si1 <- matrix(sample(seq_len(cc$Lr), n, replace = TRUE), 1, n)
  Si2 <- matrix(sample(0:1, n, replace = TRUE), 1, n)
  w.l.r <- matrix(stats::rbeta(cc$Lr, cc$a_w, cc$b_w), 1, cc$Lr)
  psi.r <- matrix(.recover_psi(stats::rbeta(cc$Lr - 1, 1, cc$a_psi_r)), 1, cc$Lr)
  xi <- matrix(0, 1, cc$Lr)
  sig2.xi.r <- rep(1, m)

  nu.alpha <- hat.alpha
  u2.alpha <- Rfast::colVars(hat.alphasij)
  a.xi.alpha <- nu.alpha
  Sij1 <- matrix(sample(seq_len(cc$L_alpha), s * J, replace = TRUE), s, J)
  Sij2 <- matrix(1, s, J)
  xi.alpha <- matrix(stats::rnorm(J * cc$L_alpha, nu.alpha, sqrt(u2.alpha)),
                     J, cc$L_alpha)
  w.alpha <- matrix(stats::rbeta(cc$L_alpha, cc$a_w_alpha, cc$b_w_alpha),
                    1, cc$L_alpha)
  psi.alpha <- matrix(.recover_psi(stats::rbeta(cc$L_alpha - 1, 1, cc$a_psi_alpha)),
                      1, cc$L_alpha)

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
  ri.muij <- matrix(ri[1, ], n, J)
  alphasij <- hat.alphasij
  alphaij <- alphasij[S, ]
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

  nsamp <- floor((niter - burnin) / thin); count.st <- 0
  F.st <- array(NA_real_, dim = c(K, Pcov, nsamp))
  Q.st <- array(NA_real_, dim = c(J, K, nsamp))
  tau.st <- matrix(0, K, nsamp)
  beta.st <- array(NA_real_, dim = c(J, Pmean, nsamp))
  ri.st <- matrix(0, nsamp, n)
  alphasij.st <- array(NA_real_, dim = c(s, J, nsamp))
  sig2.st <- numeric(nsamp)

  if (verbose) cat(sprintf("BCAIA (subject model): %d subjects, %d iters, thin %d\n",
                           s, niter, thin))
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

    ## size factor r (group-indexed, m = 1)
    RSS <- RSS + ri.muij
    pw <- update_psi_w_r(m, cc$Lr, Si1 - 1, Si2, cc$a_w, cc$b_w, cc$a_psi_r)
    psi_r_m <- abind::adrop(pw[, , 1, drop = FALSE], 3)
    w_l_r_m <- abind::adrop(pw[, , 2, drop = FALSE], 3)
    s12 <- update_Si12_grp(cc$Lr, n, m, ri, xi, cc$ur2, nu.r, w_l_r_m, psi_r_m)
    Si1 <- abind::adrop(s12[, , 1, drop = FALSE], 3) + 1
    Si2 <- abind::adrop(s12[, , 2, drop = FALSE], 3)
    xi <- update_xi_grp(m, cc$Lr, ri, Si1 - 1, Si2, a.xi, sig2.xi.r,
                        w_l_r_m, psi_r_m, nu.r, cc$ur2)
    ri <- update_ri_grp(m, n, cc$ur2, Jvec, sig2, RSS, J_bound, Si1 - 1, Si2,
                        nu.r, xi, w_l_r_m)
    ri.muij <- matrix(ri[1, ], n, J)
    RSS <- RSS - ri.muij

    ## subject-specific baseline abundance alpha_{s,j}
    pa <- update_psi_w(m, cc$L_alpha, J_bound, Sij1 - 1, Sij2,
                       cc$a_w_alpha, cc$b_w_alpha, cc$a_psi_alpha)
    psi.alpha <- abind::adrop(pa[, , 1, drop = FALSE], 3)
    w.alpha <- abind::adrop(pa[, , 2, drop = FALSE], 3)
    RSS <- RSS + alphaij
    sij <- update_Sij12(RSS, s, J, cc$L_alpha, xi.alpha, nu.alpha, w.alpha,
                        psi.alpha, Mgrp - 1, sig2, S - 1)
    Sij1 <- sij[, , 1] + 1
    Sij2 <- sij[, , 2]
    IIJ1 <- Sij1[S, ]; IIJ2 <- Sij2[S, ]
    xi.alpha <- update_xi_alpha_grp(cc$L_alpha, J, RSS, IIJ1 - 1, IIJ2, u2.alpha,
                                    a.xi.alpha, Mgrp - 1, w.alpha, sig2, nu.alpha)
    for (ii in seq_len(s)) for (j in seq_len(J))
      alphasij[ii, j] <- Sij2[ii, j] * xi.alpha[j, Sij1[ii, j]] +
        (1 - Sij2[ii, j]) *
        ((nu.alpha[j] - w.alpha[Mgrp[j], Sij1[ii, j]] * xi.alpha[j, Sij1[ii, j]]) /
           (1 - w.alpha[Mgrp[j], Sij1[ii, j]]))
    alphaij <- alphasij[S, ]
    RSS <- RSS - alphaij

    ## mean regression coefficients
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
      ri.st[count.st, ] <- ri[1, ]; alphasij.st[, , count.st] <- alphasij
      sig2.st[count.st] <- sig2
    }
    if (verbose && ni %% 1000 == 0) cat("  iter", ni, "\r")
  }
  run.time <- proc.time() - t0
  if (verbose) cat(sprintf("\nDone in %.1f min; saved %d samples.\n",
                           run.time[3] / 60, count.st))

  structure(list(
    model = "subject",
    samples = list(F = F.st, Q = Q.st, tau = tau.st, beta = beta.st,
                   ri = ri.st, alphasij = alphasij.st, sig2 = sig2.st),
    runtime = run.time,
    data = list(n = n, J = J, s = s, subject = S, Pmean = Pmean, Pcov = Pcov,
                K = K, Xmean = Xmean, Xcov = Xcov),
    control = cc, niter = niter, burnin = burnin, thin = thin, nsamp = count.st
  ), class = "bcaia")
}
