#' Empirically choose the number of latent factors K
#'
#' Applies the centred log-ratio (clr) transformation to the count matrix and
#' performs PCA on the transformed data, then returns the smallest number of
#' components whose eigenvalues explain at least \code{prop} of the total
#' variance. This is the data-driven rule for setting \code{K} described in the
#' paper.
#'
#' @param Y An \code{n x J} count matrix.
#' @param prop Proportion of variance to explain. Default \code{0.95}.
#' @param pseudocount Added before the log for the clr transform. Default
#'   \code{0.5}.
#' @param plot Logical; draw a scree plot. Default \code{FALSE}.
#'
#' @return An integer, the suggested \code{K}. The full eigenvalue vector is
#'   attached as attribute \code{"eigenvalues"}.
#' @export
#'
#' @examples
#' sim <- simulate_bcaia(seed = 1)
#' choose_K(sim$Y)
choose_K <- function(Y, prop = 0.95, pseudocount = 0.5, plot = FALSE) {
  Y <- as.matrix(Y)
  logY <- log(Y + pseudocount)
  clr <- logY - rowMeans(logY)
  ev <- eigen(stats::cov(clr), symmetric = TRUE, only.values = TRUE)$values
  ev <- pmax(ev, 0)
  K <- which(cumsum(ev) / sum(ev) >= prop)[1]
  if (plot) {
    graphics::plot(seq_along(ev), ev, type = "b", pch = 19,
                   xlab = "component", ylab = "eigenvalue",
                   main = "clr-PCA scree plot")
    graphics::abline(v = K, lty = 2, col = "red")
  }
  attr(K, "eigenvalues") <- ev
  K
}
