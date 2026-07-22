#' Hyperparameter and MCMC control settings for \code{\link{bcaia}}
#'
#' Collects the fixed hyperparameters of the model together with the
#' Metropolis-Hastings tuning constants into a single list. All values have
#' defaults matching those used in the simulation studies of Zhang, Patnode and
#' Lee; override individual entries by passing them as named arguments.
#'
#' @param a_phi,a_tau,b_tau Dirichlet-Horseshoe hyperparameters for the factor
#'   loadings. Defaults \code{a_tau = 0.1}, \code{b_tau = 1/J} (set internally
#'   from the data when \code{NULL}) and \code{a_phi = 1/20}.
#' @param a_sig,b_sig Inverse-gamma prior parameters for the idiosyncratic
#'   variance \eqn{\sigma^2}. Default \code{3} and \code{3}.
#' @param u2_beta Prior variance of the mean-regression coefficients
#'   \eqn{\beta_{jp}}. Default \code{1}.
#' @param Lr,L_alpha Stick-breaking truncation levels for the size factor
#'   \eqn{r} and baseline abundance \eqn{\alpha}. Defaults \code{30} and
#'   \code{35}.
#' @param a_psi_r,a_psi_alpha Dirichlet-process concentration parameters for
#'   \eqn{r} and \eqn{\alpha}. Default \code{1} (the subject-indexed models in
#'   the paper use \code{3}).
#' @param a_w,b_w,a_w_alpha,b_w_alpha Beta prior parameters for the inner
#'   mixture weights. Default \code{5} each.
#' @param ur2 Prior variance of the size factor \eqn{r}. Default \code{1}.
#' @param acc_tar Target acceptance rate for the adaptive Metropolis-Hastings
#'   updates. Default \code{0.234}.
#'
#' @return A named list of class \code{"bcaia_control"}.
#' @export
#'
#' @examples
#' ctrl <- bcaia_control(a_psi_r = 3, a_psi_alpha = 3)
bcaia_control <- function(a_phi = 1/20,
                          a_tau = 0.1,
                          b_tau = NULL,
                          a_sig = 3,
                          b_sig = 3,
                          u2_beta = 1,
                          Lr = 30,
                          L_alpha = 35,
                          a_psi_r = 1,
                          a_psi_alpha = 1,
                          a_w = 5,
                          b_w = 5,
                          a_w_alpha = 5,
                          b_w_alpha = 5,
                          ur2 = 1,
                          acc_tar = 0.234) {
  ctrl <- list(a_phi = a_phi, a_tau = a_tau, b_tau = b_tau,
               a_sig = a_sig, b_sig = b_sig, u2_beta = u2_beta,
               Lr = Lr, L_alpha = L_alpha,
               a_psi_r = a_psi_r, a_psi_alpha = a_psi_alpha,
               a_w = a_w, b_w = b_w,
               a_w_alpha = a_w_alpha, b_w_alpha = b_w_alpha,
               ur2 = ur2, acc_tar = acc_tar)
  class(ctrl) <- "bcaia_control"
  ctrl
}

## Internal: stick-breaking weights from Beta(1, c) draws.
.recover_psi <- function(x) {
  log_x <- log(x)
  log_x[is.infinite(log_x)] <- -.Machine$double.xmax
  res <- numeric(0)
  res[1] <- x[1]
  for (i in 2:length(x)) {
    res[i] <- exp(log_x[i] + sum(log(1 - x[1:(i - 1)])))
  }
  res[length(x) + 1] <- exp(sum(log(1 - x)))
  res
}
