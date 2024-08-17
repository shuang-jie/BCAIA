#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadilloExtensions/sample.h>
using namespace arma;


static double const log2pi = std::log(2.0 * M_PI);

void inplace_tri_mat_mult(arma::rowvec &x, arma::mat const &trimat){
  arma::uword const n = trimat.n_cols;
  
  for(unsigned j = n; j-- > 0;){
    double tmp(0.);
    for(unsigned i = 0; i <= j; ++i)
      tmp += trimat.at(i, j) * x[i];
    x[j] = tmp;
  }
}

// [[Rcpp::export]]
arma::vec dmvnrm_arma_mc(arma::mat const &x,  
                         arma::rowvec const &mean,  
                         arma::mat const &sigma, 
                         bool const logd = false) {  
  
  using arma::uword;
  uword const n = x.n_rows, 
    xdim = x.n_cols;
  mat Ik(xdim, xdim);
  arma::vec out(n);
  arma::mat const rooti = arma::inv(trimatu(arma::chol(sigma + 0.0001 * Ik)));
  double const rootisum = arma::sum(log(rooti.diag())), 
    constants = -(double)xdim/2.0 * log2pi, 
    other_terms = rootisum + constants;
  
  arma::rowvec z;
#pragma omp parallel for schedule(static) private(z)
  for (uword i = 0; i < n; i++) {
    z = (x.row(i) - mean);
    inplace_tri_mat_mult(z, rooti);   
    out(i) = other_terms - 0.5 * arma::dot(z, z);     
  }  
  
  if (logd)
    return out;
  return exp(out);
}


// [[Rcpp::export]]
double dnormLog2(arma::vec x, double means, double sds) {
  int n = x.size();
  double res=0;
  for(int i = 0; i < n; i++) {
    res += R::dnorm(x[i], means, sds, TRUE);
  }
  return res;
}

// [[Rcpp::export]]
double rgigRcpp(double lam, double chi, double psi){
  Rcpp::Environment package_env("package:GIGrvg");
  Rcpp::Function rfunction = package_env["rgig"];
  Rcpp::List test_out = rfunction(1, lam, chi, psi);
  return(test_out[0]);
}

// [[Rcpp::export]]
uvec seq_cpp(int locppid, int hicppid) {
  int n = hicppid - locppid + 1;
  uvec sequence(n);
  for(int i = 0; i < n; i++) {
    sequence[i] = locppid + i;
  }
  return sequence;
}

// [[Rcpp::export]]
arma::mat mvrnormArma(int n, arma::vec mu, arma::mat sigma) {
  int ncols = sigma.n_cols;
  arma::mat YY = arma::randn(n, ncols);
  return arma::repmat(mu, 1, n).t() + YY * arma::chol(sigma);
}

// [[Rcpp::export]]
mat cpp_inv(mat A) {
  return A.i();
}

// [[Rcpp::export]]
mat update_ystar(mat logY, mat logY1, int n, int J, mat RSS, double sig2){
  mat y_star(n, J);
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < J; j++) {
      double u_min = R::pnorm(logY(i,j), RSS(i,j), sqrt(sig2), 1, 0);
      double u_max = R::pnorm(logY1(i,j), RSS(i,j), sqrt(sig2), 1, 0);
      double uij = R::runif(u_min, u_max);
      if(uij == 1) {uij = 1-9.9*pow(10, -17);}
      if(uij == 0) {uij = 9.9*pow(10, -17);}
      y_star(i,j) = R::qnorm(uij, RSS(i,j), sqrt(sig2), 1, 0);
    }
  }
  return(y_star);
}

// [[Rcpp::export]]
double update_sig2(int n, int J, mat RSS, double a_sig, double b_sig){
  double res;
  res = 1.0/ (R::rgamma(a_sig + n * J * 0.5, 1.0/(b_sig+ 0.5 * sum(sum(RSS % RSS))) ) );
  return(res);
}

// [[Rcpp::export]]
mat update_Q(int J, int K, mat zeta, rowvec tau, mat phi, mat eeta, mat RSS, double sig2){
  mat res(J, K);
  for(int j = 0; j < J; j++) {
    rowvec Dj = zeta.row(j) % phi.row(j) % tau;
    Dj = 1.0/Dj;
    arma::uvec idD0 = find(abs(Dj) > pow(10, 17));
    Dj.elem(idD0).fill(pow(10, 17));
    mat inv_lam = (diagmat(Dj) + (eeta.t() * eeta)/sig2 ).i();
    vec mu_lam = inv_lam * eeta.t() * RSS.col(j) / sig2 ;
    rowvec ttt = mvrnormArma(1, mu_lam,  inv_lam);
    arma::uvec idx0 = find(abs(ttt) < pow(10, -100));
    ttt.elem(idx0).fill(pow(10, -100));
    res.row(j) = ttt;
  }
  return(res);
}

// [[Rcpp::export]]
cube update_Lami(mat Q, mat F, mat X, int J, int K, int n){
  cube res(J, K, n);
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < J; j++) {
      for(int k = 0; k < K; k++) {
        res(j, k, i) = Q(j, k) * (sum(X.row(i) % F.row(k)));
      }
    }
  }
  return(res);
}

// [[Rcpp::export]]
cube update_ZZ_ZZeta(int J, int K, mat LLambda, mat pphi_m, rowvec ttau, mat ZZ, mat zzeta){
  for(int j = 0; j < J; j++) {
    for(int k = 0; k < K; k++) {
      zzeta(j, k) = 1.0/ (R::rgamma(1.0, 1.0/( 1.0/ZZ(j,k) + pow(LLambda(j,k), 2) /(2.0*pphi_m(j,k)*ttau(k))) ));
      ZZ(j, k) = 1.0/ (R::rgamma(1.0, 1.0/(1.0 + 1.0/zzeta(j, k)) ));
    }
  }
  cube res(J, K, 2);
  res.slice(0) = zzeta;
  res.slice(1) = ZZ;
  return(res);
}

// [[Rcpp::export]]
double pos_phi(vec t_p, int wj, int wk, int Jsum, double a_phi, mat Lambda, mat zeta, 
               rowvec tau) {
  double res = R::dgamma(t_p(wj), a_phi, 1, 1) + log(t_p(wj));
  vec phi = t_p/ sum(t_p);
  arma::uvec idx0 = find(phi == 0);
  arma::uvec idx = find(phi != 0);
  double fi = min(phi.elem(idx));
  phi.elem(idx0).fill(fi);
  for(int jj = 0; jj < Jsum; jj++){
    double hssd = sqrt(zeta(jj, wk) *  phi(jj) * tau(wk) );
    res = res + R::dnorm(Lambda(jj, wk), 0, hssd, 1);
  }
  return(res);
}

// [[Rcpp::export]]
cube update_phi(int K, int Jsum, mat til_phi_m, mat sig_pro,
                double a_phi, mat Lambda, mat zeta, rowvec tau, mat acc,
                mat phi_m, mat nacc, mat true_nacc, int ni, double acc_tar, 
                double delta){
  double ind;
  for(int k = 0; k < K; k++) {
    for(int j = 0; j < Jsum; j++) {
      mat til_phi_pro = til_phi_m;
      double new_pro = R::rnorm(0, 1) * sig_pro(j, k) + log(til_phi_m(j, k));
      til_phi_pro(j, k) = exp(new_pro);
      double d_log = pos_phi(til_phi_pro.col(k), j, k, Jsum, a_phi, Lambda, zeta, tau) -
        pos_phi(til_phi_m.col(k), j, k, Jsum, a_phi, Lambda, zeta, tau);
      vec tt(2); tt(0)=1; tt(1) = exp(d_log); acc(j, k) = min(tt);
      double cc = R::runif(0, 1);
      if(cc<acc(j, k)){
        til_phi_m(j, k) = exp(new_pro);
        vec ttt = til_phi_m.col(k) / sum(til_phi_m.col(k));
        arma::uvec idx = find(ttt < pow(10, -100));
        ttt.elem(idx).fill(pow(10, -100));
        phi_m.col(k) = ttt;
        nacc(j, k) ++;
        true_nacc(j, k) ++;
      }
      
      if(ni % 50 == 0){
        if(nacc(j, k)/50>acc_tar){
          ind = 1;
        } else{
          ind = -1;
        }
        sig_pro(j, k) = sig_pro(j, k) * exp(delta * ind);
        nacc(j, k) = 0;
      }
    }
  }
  
  cube res(Jsum, K, 5);
  res.slice(0) = til_phi_m;
  res.slice(1) = phi_m;
  res.slice(2) = nacc;
  res.slice(3) = true_nacc;
  res.slice(4) = sig_pro;
  return(res);
}

// [[Rcpp::export]]
rowvec update_tau_GIG(int K, double a_tau, double b_tau, int Jsum, mat Lambda, mat phi_m, mat zeta){
  rowvec rreess(K);
  for(int k = 0; k < K; k++) {
    double par1 = a_tau - Jsum/2.0;
    double par2 = sum( pow(Lambda.col(k), 2)  / (phi_m.col(k) % zeta.col(k)) );
    double par3 = 2.0 * b_tau;
    rreess(k) = rgigRcpp(par1, par2, par3);
  }
  return(rreess);
}

// [[Rcpp::export]]
double pos_F_k(int n, int J, mat RSS, cube Lambdai, double sig2, vec Fk) {
  mat Ik; Ik.eye(J, J);
  double res = 0; mat Sigma(J, J);
  rowvec me(J, fill::zeros);
  for(int i = 0; i < n; i++) {
    Sigma = sig2 * Ik + Lambdai.slice(i) * Lambdai.slice(i).t();
    res += dmvnrm_arma_mc(RSS.row(i),  me, Sigma, true)[0];
  }
  res = res + dnormLog2(Fk, 0, 1);
  return(res);
}

// [[Rcpp::export]]
double pos_F_kp(int n, int J, mat RSS, cube Lambdai, double sig2, double Fkp) {
  mat Ik; Ik.eye(J, J);
  double res = 0; mat Sigma(J, J);
  rowvec me(J, fill::zeros);
  for(int i = 0; i < n; i++) {
    Sigma = sig2 * Ik + Lambdai.slice(i) * Lambdai.slice(i).t();
    res += dmvnrm_arma_mc(RSS.row(i),  me, Sigma, true)[0];
  }
  res = res + R::dnorm(Fkp, 0, 1, 1);
  return(res);
}

// [[Rcpp::export]]
double pos_F_kp0(int n, int J, mat RSS, cube Lambdai, double sig2, double Fkp) {
  mat Ik; Ik.eye(J, J);
  if(Fkp<0){
    double res;
    res = -pow(10, 10);
    return(res);
  } else {
    double res = 0; mat Sigma(J, J);
    rowvec me(J, fill::zeros);
    for(int i = 0; i < n; i++) {
      Sigma = sig2 * Ik + Lambdai.slice(i) * Lambdai.slice(i).t();
      res += dmvnrm_arma_mc(RSS.row(i),  me, Sigma, true)[0];
    }
    res = res + 2 * R::dnorm(Fkp, 0, 1, 1);
    return(res);
  }
}


// [[Rcpp::export]]
Rcpp::List update_F_k_MCMC_signswitching(int n, int J, int K, int P, mat F, mat Q, mat X, double sig2, mat RSS,
                                    rowvec nacc_Fk, double acc_tar, int ni, double gamma_adap, 
                                    rowvec loglambda_adap, mat mu_adap, cube S_adap){
  double ind; mat Fnew(K, P, fill::zeros); mat pro_S(J, J); 
  double d_log; double acc; double cc; 
  rowvec new_Fkp;
  rowvec a_mu_diff;
  cube Lambdaiold(J, K, n); cube Lambdainew(J, K, n);
  
  for(int k = 0; k < K; k++) {
    Fnew = F; 
    pro_S = exp(loglambda_adap(k)/2.0) * S_adap.slice(k);
    new_Fkp = mvrnormArma(1, F.row(k).t(),  pro_S);
    Fnew.row(k) = new_Fkp;
    Lambdaiold = update_Lami(Q, F, X, J, K, n);
    Lambdainew = update_Lami(Q, Fnew, X, J, K, n);
    d_log = pos_F_k(n, J, RSS, Lambdainew, sig2, new_Fkp.t()) -
      pos_F_k(n, J, RSS, Lambdaiold, sig2, F.row(k).t());
    d_log = 0;
    vec tt(2); tt(0)=1; tt(1) = exp(d_log); acc = min(tt);
    cc = R::runif(0, 1);
    
    if(cc<acc){
      F = Fnew;
      nacc_Fk(k) ++;
    }
    
    loglambda_adap(k) = loglambda_adap(k) + gamma_adap * (acc - acc_tar);
    a_mu_diff = F.row(k) - mu_adap.row(k);
    mu_adap.row(k) = mu_adap.row(k) + gamma_adap * a_mu_diff;
    S_adap.slice(k) = S_adap.slice(k) + gamma_adap*( a_mu_diff.t() * a_mu_diff - S_adap.slice(k) );
  }
  
  Rcpp::List res;
  res["F"] = F;
  res["mu_adap_Fk"] = mu_adap;
  res["nacc_Fk"] = nacc_Fk;
  res["Sigma_adap_Fk"] = S_adap;
  res["loglambda_adap_Fk"] = loglambda_adap;
  return(res);
}



// [[Rcpp::export]]
cube update_F_kp_MCMC_signswitching(int n, int J, int K, int P, mat F, mat Q, mat X, mat sig_pro_Fkp , double sig2, mat RSS,
                                    mat nacc_Fkp, mat true_nacc_Fkp, double acc_tar, int ni, double gamma_adap, 
                                    mat loglambda_adap, mat mu_adap){
  double ind;
  cube Lambdaiold(J, K, n); cube Lambdainew(J, K, n);
  for(int k = 0; k < K; k++) {
    for(int p = 0; p < P; p++) {
      mat Fnew = F;
      double new_Fkp = R::rnorm(0, 1) * sig_pro_Fkp(k, p) * exp(loglambda_adap(k,p)/2.0) + F(k, p);
      Fnew(k, p) = new_Fkp;
      Lambdaiold = update_Lami(Q, F, X, J, K, n);
      Lambdainew = update_Lami(Q, Fnew, X, J, K, n);
      double d_log = pos_F_kp(n, J, RSS, Lambdainew, sig2, new_Fkp) - 
        pos_F_kp(n, J, RSS, Lambdaiold, sig2, F(k, p));
      
      vec tt(2); tt(0)=1; tt(1) = exp(d_log); double acc = min(tt);
      double cc = R::runif(0, 1);
      
      if(cc<acc){
        F = Fnew;
        nacc_Fkp(k, p) ++;
        true_nacc_Fkp(k, p) ++;
      }
      
      loglambda_adap(k, p) = loglambda_adap(k, p) + gamma_adap * (acc - acc_tar);
      double a_mu_diff = F(k, p) - mu_adap(k, p);
      mu_adap(k, p) = mu_adap(k, p) + gamma_adap * a_mu_diff;
      sig_pro_Fkp(k, p) = sig_pro_Fkp(k, p) + gamma_adap*(pow(a_mu_diff, 2) - sig_pro_Fkp(k, p));
    }
  }
  
  cube res(K, P, 6);
  res.slice(0) = F;
  res.slice(1) = mu_adap;
  res.slice(2) = nacc_Fkp;
  res.slice(3) = true_nacc_Fkp;
  res.slice(4) = sig_pro_Fkp;
  res.slice(5) = loglambda_adap;
  return(res);
}


// [[Rcpp::export]]
cube update_F_kp_MCMC_50(int n, int J, int K, int P, mat F, mat Q, mat X, mat sig_pro_Fkp , double sig2, mat RSS,
                      mat nacc_Fkp, mat true_nacc_Fkp, double acc_tar, int ni, double gamma_adap, 
                      mat loglambda_adap, mat mu_adap){
  double ind; double new_Fkp; double d_log; double cc; double acc;
  cube Lambdaiold(J, K, n); cube Lambdainew(J, K, n);
  for(int k = 0; k < K; k++) {
    for(int p = 0; p < P; p++) {
        mat Fnew = F;
        new_Fkp = R::rnorm(0, 1) * sig_pro_Fkp(k, p) + F(k, p);
        Fnew(k, p) = new_Fkp;
        Lambdaiold = update_Lami(Q, F, X, J, K, n);
        Lambdainew = update_Lami(Q, Fnew, X, J, K, n);
        d_log = pos_F_kp0(n, J, RSS, Lambdainew, sig2, new_Fkp) - pos_F_kp0(n, J, RSS, Lambdaiold, sig2, F(k, p));
        vec tt(2); tt(0)=1; tt(1) = exp(d_log); acc = min(tt);
        cc = R::runif(0, 1);
        if(cc<acc){
          F = Fnew;
          nacc_Fkp(k, p) ++;
          true_nacc_Fkp(k, p) ++;
        }
        if(ni % 50 == 0){
          if(nacc_Fkp(k, p)/50>acc_tar){
            ind = 1;
          } else{
            ind = -1;
          }
          sig_pro_Fkp(k, p) = sig_pro_Fkp(k, p) * exp(gamma_adap * ind);
          nacc_Fkp(k, p) = 0;
        }
      }
    }
  
  cube res(K, P, 6);
  res.slice(0) = F;
  res.slice(1) = mu_adap;
  res.slice(2) = nacc_Fkp;
  res.slice(3) = true_nacc_Fkp;
  res.slice(4) = sig_pro_Fkp;
  res.slice(5) = loglambda_adap;
  return(res);
}


// [[Rcpp::export]]
cube update_F_kp_MCMC(int n, int J, int K, int P, mat F, mat Q, mat X, mat sig_pro_Fkp , double sig2, mat RSS,
                      mat nacc_Fkp, mat true_nacc_Fkp, double acc_tar, int ni, double gamma_adap, 
                      mat loglambda_adap, mat mu_adap){
  double ind;
  cube Lambdaiold(J, K, n); cube Lambdainew(J, K, n);
  for(int k = 0; k < K; k++) {
    for(int p = 0; p < P; p++) {
      
      if(p == 0){
        mat Fnew = F;
        double new_Fkp = R::rnorm(0, 1) * sig_pro_Fkp(k, p) + F(k, p);
        Fnew(k, p) = new_Fkp;
        Lambdaiold = update_Lami(Q, F, X, J, K, n);
        Lambdainew = update_Lami(Q, Fnew, X, J, K, n);
        
        double d_log = pos_F_kp0(n, J, RSS, Lambdainew, sig2, new_Fkp) - 
          pos_F_kp0(n, J, RSS, Lambdaiold, sig2, F(k, p));
        
        vec tt(2); tt(0)=1; tt(1) = exp(d_log); double acc = min(tt);
        double cc = R::runif(0, 1);
        
        if(cc<acc){
          F = Fnew;
          nacc_Fkp(k, p) ++;
          true_nacc_Fkp(k, p) ++;
        }
        
        if(ni % 50 == 0){
          if(nacc_Fkp(k, p)/50>acc_tar){
            ind = 1;
          } else{
            ind = -1;
          }
          sig_pro_Fkp(k, p) = sig_pro_Fkp(k, p) * exp(gamma_adap * ind);
          nacc_Fkp(k, p) = 0;
        }
        
      } else {
        mat Fnew = F;
        double new_Fkp = R::rnorm(0, 1) * sig_pro_Fkp(k, p) * exp(loglambda_adap(k,p)/2.0) + F(k, p);
        Fnew(k, p) = new_Fkp;
        Lambdaiold = update_Lami(Q, F, X, J, K, n);
        Lambdainew = update_Lami(Q, Fnew, X, J, K, n);
        double d_log = pos_F_kp(n, J, RSS, Lambdainew, sig2, new_Fkp) - 
          pos_F_kp(n, J, RSS, Lambdaiold, sig2, F(k, p));
        
        vec tt(2); tt(0)=1; tt(1) = exp(d_log); double acc = min(tt);
        double cc = R::runif(0, 1);
        
        if(cc<acc){
          F = Fnew;
          nacc_Fkp(k, p) ++;
          true_nacc_Fkp(k, p) ++;
        }
        
        loglambda_adap(k, p) = loglambda_adap(k, p) + gamma_adap * (acc - acc_tar);
        double a_mu_diff = F(k, p) - mu_adap(k, p);
        mu_adap(k, p) = mu_adap(k, p) + gamma_adap * a_mu_diff;
        sig_pro_Fkp(k, p) = sig_pro_Fkp(k, p) + gamma_adap*(pow(a_mu_diff, 2) - sig_pro_Fkp(k, p));
        
      }
      
      
      
    }
  }
  
  cube res(K, P, 6);
  res.slice(0) = F;
  res.slice(1) = mu_adap;
  res.slice(2) = nacc_Fkp;
  res.slice(3) = true_nacc_Fkp;
  res.slice(4) = sig_pro_Fkp;
  res.slice(5) = loglambda_adap;
  return(res);
}

// [[Rcpp::export]]
mat update_RSS(mat ystar, cube Lambdaiarray, mat eta, int n, int J, mat rij, mat alphaij){
  mat res(n, J);
  mat lam;
  for(int i = 0; i < n; i++) {
    lam = Lambdaiarray.slice(i);
    for(int j = 0; j < J; j++) {
      res(i, j) = ystar(i, j) - sum(eta.row(i) % lam.row(j)) - rij(i, j)- alphaij(i,j);
    }
  }
  return(res);
}

// [[Rcpp::export]]
mat update_F_kp_2(int n, int J, int K, int P, mat Q, mat X, mat eta, double sig2, mat RSS, mat F){
  mat epsij(n, J); mat cij(n, J);
  for(int k = 0; k < K; k++) {
    for(int p = 0; p < P; p++) {
      
      for(int i = 0; i < n; i++) {
        for(int j = 0; j < J; j++) {
          rowvec qj = Q.row(j);
          rowvec etai = eta.row(i);
          rowvec FX= X.row(i) * F.t();
          
          epsij(i,j) = RSS(i,j) - sum(qj % etai % FX) + eta(i,k) * Q(j,k) * F(k,p) *X(i,p) ;
          cij(i,j) = eta(i,k) * Q(j,k) * X(i, p);
        }
      }
      double varF = 1.0/(1.0+ sum(sum(cij % cij))/ sig2);
      double muF = varF * sum(sum(epsij % cij)) / sig2;
      F(k, p) = R::rnorm(muF, sqrt(varF));
      
    }
  }
  return(F);
}


// [[Rcpp::export]]
mat update_F_kp_3(int n, int J, int K, int P, mat Q, mat X, mat eta, double sig2, mat ystar, mat F,
                  mat rij, mat alphaij){
  mat res = F;
  mat epsij(n, J); mat cij(n, J);
  mat eps(n, J); cube Lambdaiarray(J, K, n);
  for(int k = 0; k < K; k++) {
    for(int p = 0; p < P; p++) {
      
      Lambdaiarray = update_Lami(Q, res, X, J, K, n);
      eps = update_RSS(ystar, Lambdaiarray, eta, n, J, rij, alphaij);
      
      for(int i = 0; i < n; i++) {
        for(int j = 0; j < J; j++){
          epsij(i,j) = eps(i,j) + eta(i,k) * Q(j,k) * res(k,p) *X(i,p) ;
          cij(i,j) = eta(i,k) * Q(j,k) * X(i, p);
        }
      }
      double varF = 1.0/(1.0+ sum(sum(cij % cij))/ sig2);
      double muF = varF * sum(sum(epsij % cij)) / sig2;
      res(k, p) = R::rnorm(muF, sqrt(varF));
    }
  }
  return(res);
}


// [[Rcpp::export]]
mat update_F_kp_3_no0(int n, int J, int K, int P, mat Q, mat X, mat eta, double sig2, mat ystar, mat F,
                      mat rij, mat alphaij){
  mat res = F;
  mat epsij(n, J); mat cij(n, J);
  mat eps(n, J); cube Lambdaiarray(J, K, n);
  for(int k = 0; k < K; k++) {
    for(int p = 1; p < P; p++) {
      
      Lambdaiarray = update_Lami(Q, res, X, J, K, n);
      eps = update_RSS(ystar, Lambdaiarray, eta, n, J, rij, alphaij);
      
      for(int i = 0; i < n; i++) {
        for(int j = 0; j < J; j++){
          epsij(i,j) = eps(i,j) + eta(i,k) * Q(j,k) * res(k,p) *X(i,p) ;
          cij(i,j) = eta(i,k) * Q(j,k) * X(i, p);
        }
      }
      double varF = 1.0/(1.0+ sum(sum(cij % cij))/ sig2);
      double muF = varF * sum(sum(epsij % cij)) / sig2;
      res(k, p) = R::rnorm(muF, sqrt(varF));
    }
  }
  return(res);
}


// [[Rcpp::export]]
mat update_Delta(int K, cube Lambdai, double sig2, int n, mat RSS){
  mat Delta(n, K); 
  mat Ik; Ik.eye(K, K);
  for(int i = 0; i < n; i++) {
    mat inv_e = (Ik + (Lambdai.slice(i).t() * Lambdai.slice(i))/sig2).i();
    vec mu_e = inv_e * (Lambdai.slice(i)).t() * (RSS.row(i)).t() /sig2;
    Delta.row(i) = mvrnormArma(1, mu_e,  inv_e);
  }
  return(Delta);
}

// [[Rcpp::export]]
mat update_Si12(int Lr, int n, rowvec ri, rowvec xi, double ur2, double nu_r, rowvec w_l_r, rowvec psi_r){
  vec piil1(Lr), piil0(Lr);
  rowvec Si1(n);
  rowvec Si2(n);
    for(int i = 0; i < n; i++) {
      for(int l = 0; l< Lr; l++){
        piil1(l)=R::dnorm(ri(i), xi(l), sqrt(ur2), 1) + log(w_l_r(l)) + log(psi_r(l));
        piil0(l)=R::dnorm(ri(i), (nu_r-w_l_r(l)*xi(l) )/(1-w_l_r(l)) , sqrt(ur2), 1) + log(1-w_l_r(l)) + log(psi_r(l));
      }
      vec piil = join_cols(piil1,piil0);
      piil = exp(piil - max(piil));
      piil = piil/sum(piil);
      rowvec try4 = linspace<rowvec>(0, 2*Lr-1, 2*Lr);
      int id = Rcpp::RcppArmadillo::sample(try4, 1, false, piil)(0);
      
      if(id<Lr){
        Si1(i) = id;
        Si2(i) = 1;
      } else {
        Si1(i) = id - Lr;
        Si2(i) = 0;
      }
    }
  mat res(n, 2);
  res.col(0) = Si1.t();
  res.col(1) = Si2.t();
  return(res);
}

// [[Rcpp::export]]
rowvec update_xi(int Lr, rowvec ri, rowvec Si1, rowvec Si2, double a_xi, double sig2_xi_r,
              rowvec w_l_r, double nu_r, double ur2){
  rowvec xi(Lr);
    for(int l=0; l< Lr; l++){
      int sumSi1 = sum(Si1==l);
      int sumSi1Si20 = sum((Si1==l) % (Si2 ==0));
      int sumSi1Si21 = sum((Si1==l) % (Si2 ==1));
      
      if (sumSi1== 0){
        xi(l) = R::rnorm(a_xi, sqrt(sig2_xi_r));
      } else {
        double sci2 = sumSi1Si21 + sumSi1Si20 * pow(w_l_r(l)/(1-w_l_r(l)), 2);
        arma::uvec idx1 = find( (Si1==l)  % (Si2 ==1) );
        arma::uvec idx0 = find( (Si1==l)  % (Si2 ==0) );
        double tr = sum( ri.elem(idx1) ) -w_l_r(l)/(1-w_l_r(l)) * sum( ri.elem(idx0) - nu_r/(1-w_l_r(l)) );
        double p_var = 1.0/(1.0/sig2_xi_r+sci2/ur2);
        double p_m = (a_xi/sig2_xi_r + tr/ur2) * p_var;
        xi(l) = R::rnorm(p_m, sqrt(p_var));
      }
    }
  return(xi);
}

// [[Rcpp::export]]
rowvec update_ri(int n, double ur2, double J, double sig2, mat RSS, 
                 rowvec Si1, rowvec Si2, double nu_r, rowvec xi, rowvec w_l_r){
  rowvec ri(n);
  double pos_r_var = 1.0/(1.0/ur2 + J/sig2);
  for(int i = 0; i < n; i++) {
    double prior_r_m = Si2(i)  *  xi(Si1(i)) + (1 - Si2(i)) *
      (nu_r-w_l_r(Si1(i)) * xi(Si1(i)) )/(1-w_l_r(Si1(i)));
    double pos_r_m = pos_r_var * (prior_r_m/ur2+ sum(RSS.row(i))/sig2);
    ri(i) =  R::rnorm(pos_r_m, sqrt(pos_r_var));
  }
  return(ri);
}

// [[Rcpp::export]]
double poswrt(double winput, rowvec riinput, double sumSi1lSi21, double sumSi1lSi20, double sig2,
             double a_w_r, double b_w_r, double xi_l_rinput, double nu_r){
  double res;
  rowvec rifinal = riinput - (nu_r-winput*xi_l_rinput)/(1-winput);
  double rrss = sum(sum(rifinal % rifinal));
  res= log(winput) * (a_w_r + sumSi1lSi21  +1) + 
    log(1-winput) * (b_w_r + sumSi1lSi20  +1) - 1.0/(2.0*sig2) * rrss;
  return(res);
}

// // [[Rcpp::export]]
// rowvec update_w_r(rowvec w_r, int Lr, rowvec Si1, rowvec Si2, double a_w_r, double b_w_r,
//                   rowvec sig_pro_wlr, mat RSS, rowvec loglbd_wr, double sig2, rowvec xi_l_r, double nu_r,
//                   rowvec nacc_w_r, rowvec true_nacc_w_r, double gamma_adap, double acc_tar,
//                   rowvec mu_adap_w_r){
//   for(int l=0; l< Lr; l++){
//     double old_w = w_r(l);
//     double old_tw = log(w_r(l)/(1-w_r(l)));
//     double new_tw = R::rnorm(0, 1) * sig_pro_wlr(l) * exp(loglbd_wr(l)/2.0) + old_tw;
//     double new_w = exp(new_tw)/(1+exp(new_tw));
//     
//     arma::uvec idx0 = find( (Si1==l)  % (Si2==0) );
//     mat RSS_sub = RSS.rows(idx0);
//     
//     double sumSjlSj21 = sum( (Si1==l) % (Si2 ==1));
//     double sumSjlSj20 = sum( (Si1==l) % (Si2 ==0));
//     double xi_l_ainput = xi_l_r(l);
//     
//     double d_log =  poswt(new_w, RSS_sub, sumSjlSj21, sumSjlSj20, sig2, a_w_r, b_w_r, xi_l_ainput, nu_r)
//       - poswt(old_w, RSS_sub, sumSjlSj21, sumSjlSj20, sig2, a_w_r, b_w_r, xi_l_ainput, nu_r);
//     vec tt(2); tt(0)=1; tt(1) = exp(d_log); double acc = min(tt);
//     double cc = R::runif(0, 1);
//     
//     if(cc<acc){
//       w_r(l) = new_w;
//       nacc_w_r(l) ++;
//       true_nacc_w_r(l) ++;
//     }
//     
//     loglbd_wr(l) = loglbd_wr(l) + gamma_adap * (acc - acc_tar);
//     double w_a_mu_diff =  w_r(l)  - mu_adap_w_r(l);
//     mu_adap_w_r(l) = mu_adap_w_r(l) + gamma_adap * w_a_mu_diff;
//     sig_pro_wlr(l) = sig_pro_wlr(l) + gamma_adap*(pow(w_a_mu_diff, 2) - sig_pro_wlr(l));
//   }
//   return(w_r);
// }

// [[Rcpp::export]]
mat update_w_r_new(rowvec w_r, int Lr, rowvec Si1, rowvec Si2, double a_w_r, double b_w_r,
                  rowvec sig_pro_wlr, rowvec ri, rowvec loglbd_wr, double sig2, rowvec xi_l_r, double nu_r,
                  rowvec nacc_w_r, rowvec true_nacc_w_r, double gamma_adap, double acc_tar,
                  rowvec mu_adap_w_r, int n){
  for(int l=0; l< Lr; l++){
    double sumSi1lSi21 = sum( (Si1==l) % (Si2 ==1));
    double sumSi1lSi20 = sum( (Si1==l) % (Si2 ==0));
    
    if(sumSi1lSi20 ==0){
      w_r(l) = R::rbeta(a_w_r + sumSi1lSi21, b_w_r);
    } else{
    double old_w = w_r(l);
    double old_tw = log(w_r(l)/(1-w_r(l)));
    double new_tw = R::rnorm(0, 1) * sig_pro_wlr(l) * exp(loglbd_wr(l)/2.0) + old_tw;
    double new_w = exp(new_tw)/(1+exp(new_tw));
    uvec idx0 = find( (Si1==l)  % (Si2==0) );
    rowvec  ri_sub = (ri.elem(idx0)).t();
    
    double xi_l_ainput = xi_l_r(l);

    double d_log =  poswrt(new_w, ri_sub, sumSi1lSi21, sumSi1lSi20, sig2, a_w_r, b_w_r, xi_l_ainput, nu_r)
      - poswrt(old_w, ri_sub, sumSi1lSi21, sumSi1lSi20, sig2, a_w_r, b_w_r, xi_l_ainput, nu_r);

    vec tt(2); tt(0)=1; tt(1) = exp(d_log); double acc = min(tt);
    double cc = R::runif(0, 1);

    if(cc<acc){
      w_r(l) = new_w;
      nacc_w_r(l) ++;
      true_nacc_w_r(l) ++;
    }

    loglbd_wr(l) = loglbd_wr(l) + gamma_adap * (acc - acc_tar);
    double w_a_mu_diff =  w_r(l)  - mu_adap_w_r(l);
    mu_adap_w_r(l) = mu_adap_w_r(l) + gamma_adap * w_a_mu_diff;
    sig_pro_wlr(l) = sig_pro_wlr(l) + gamma_adap*(pow(w_a_mu_diff, 2) - sig_pro_wlr(l));
    }
  }
  mat res(Lr, 3);
  res.col(0) = w_r.t();
  res.col(1) = mu_adap_w_r.t();
  res.col(2) = sig_pro_wlr.t();
  return(res);
}

// [[Rcpp::export]]
rowvec update_psi_a(int Lalpha, rowvec Sj1, rowvec Sj2, double a_psi_alpha){
  
  rowvec psi_a(Lalpha);
  rowvec M_La(Lalpha);
  rowvec V_a(Lalpha);
  
  for(int l=0; l< Lalpha; l++){
    M_La(l) = sum(Sj1==l);
  }
  
  for(int l=0; l< Lalpha; l++){
    uvec pos = seq_cpp(l+1, Lalpha-1);
    V_a(l) = R::rbeta(1+M_La(l), a_psi_alpha + sum(M_La.elem(pos)));
  }
  
  rowvec try1 = V_a; try1(Lalpha-1) = 1;
  rowvec try2(Lalpha); try2(0)=1;
  
  for(int l=1; l< Lalpha; l++){
    try2(l) = 1-V_a(l-1);
  }
  try2 = cumprod(try2);
  psi_a = try1 % try2;
  
  return(psi_a);
}


// [[Rcpp::export]]
double poswt(double winput, mat RSSinput, double sumSjlSj21, double sumSjlSj20, double sig2,
             double a_w_alpha, double b_w_alpha, double xi_l_ainput, double nu_a){
  double res;
  mat RSSfinal = RSSinput - (nu_a-winput*xi_l_ainput)/(1-winput);
  double rrss = sum(sum(RSSfinal % RSSfinal));
  res= log(winput) * (a_w_alpha + sumSjlSj21  +1) + 
    log(1-winput) * (b_w_alpha + sumSjlSj20  +1) - 1.0/(2.0*sig2) * rrss;
  return(res);
}

// [[Rcpp::export]]
mat update_w_a(rowvec w_a, int Lalpha, rowvec Sj1, rowvec Sj2, double a_w_alpha, double b_w_alpha,
                  rowvec sig_pro_wla, mat RSS, rowvec loglbd_wa, double sig2, rowvec xi_l_a, double nu_a,
                  rowvec nacc_w_a, rowvec true_nacc_w_a, double gamma_adap, double acc_tar,
                  rowvec mu_adap_w_a){
  for(int l=0; l< Lalpha; l++){
    double sumSjlSj21 = sum( (Sj1==l) % (Sj2 ==1));
    double sumSjlSj20 = sum( (Sj1==l) % (Sj2 ==0));
    
    if(sumSjlSj20 == 0){
      w_a(l) = R::rbeta(a_w_alpha + sumSjlSj21, b_w_alpha);
    } else{
      
      double old_w = w_a(l);
      double old_tw = log(w_a(l)/(1-w_a(l)));
      double new_tw = R::rnorm(0, 1) * sig_pro_wla(l) * exp(loglbd_wa(l)/2.0) + old_tw;
      double new_w = exp(new_tw)/(1+exp(new_tw));
      
      arma::uvec idx0 = find( (Sj1==l)  % (Sj2==0) );
      mat RSS_sub = RSS.cols(idx0);
      
      double xi_l_ainput = xi_l_a(l);
      
      double d_log =  poswt(new_w, RSS_sub, sumSjlSj21, sumSjlSj20, sig2, a_w_alpha, b_w_alpha, xi_l_ainput, nu_a)
        - poswt(old_w, RSS_sub, sumSjlSj21, sumSjlSj20, sig2, a_w_alpha, b_w_alpha, xi_l_ainput, nu_a);
      vec tt(2); tt(0)=1; tt(1) = exp(d_log); double acc = min(tt);
      double cc = R::runif(0, 1);
      
      if(cc<acc){
        w_a(l) = new_w;
        nacc_w_a(l) ++;
        true_nacc_w_a(l) ++;
      }
      
      loglbd_wa(l) = loglbd_wa(l) + gamma_adap * (acc - acc_tar);
      double w_a_mu_diff =  w_a(l)  - mu_adap_w_a(l);
      mu_adap_w_a(l) = mu_adap_w_a(l) + gamma_adap * w_a_mu_diff;
      sig_pro_wla(l) = sig_pro_wla(l) + gamma_adap*(pow(w_a_mu_diff, 2) - sig_pro_wla(l));
    }
  }
  mat res(Lalpha, 3);
  res.col(0) = w_a.t();
  res.col(1) = mu_adap_w_a.t();
  res.col(2) = sig_pro_wla.t();
  return(res);
}

// [[Rcpp::export]]
mat update_Sj12(mat RSS,  int J, int Lalpha, rowvec xi_alpha, double nu_alpha, rowvec w_l_a, rowvec psi_a,
                  double sig2){
  rowvec Sj1(J);
  rowvec Sj2(J);

    for(int j = 0; j < J; j++) {
      vec pjl1(Lalpha), pjl0(Lalpha);

      for(int l = 0; l< Lalpha; l++){
        pjl1(l) = dnormLog2(RSS.col(j), xi_alpha(l), sqrt(sig2)) + log(w_l_a(l)) + log(psi_a(l));
        pjl0(l) = dnormLog2(RSS.col(j), (nu_alpha - w_l_a(l) * xi_alpha(l))/(1-w_l_a(l)), sqrt(sig2)) + log(1-w_l_a(l)) + log(psi_a(l));
      }

      vec piil = join_cols(pjl1, pjl0);
      piil = exp(piil - max(piil));
      piil = piil/sum(piil);
      rowvec try4 = linspace<rowvec>(0, 2*Lalpha-1, 2*Lalpha);
      vec id = Rcpp::RcppArmadillo::sample(try4, 1, false, piil);

      if(id(0)<Lalpha){
        Sj1(j) = id(0);
        Sj2(j) = 1;
      } else {
        Sj1(j) = id(0) - Lalpha;
        Sj2(j) = 0;
      }
    }

  mat res(J, 2);
  res.col(0) = Sj1.t();
  res.col(1) = Sj2.t();
  return(res);
}

// [[Rcpp::export]]
rowvec update_xi_alpha(int Lalpha, int n, int J, mat RSS, rowvec Sj1, rowvec Sj2,
                    double u2_alpha, double a_xi_alpha, rowvec w_l_a, double sig2, double nu_alpha){
  rowvec res(Lalpha);
  for(int l=0; l< Lalpha; l++){
      int sumSj1 = sum(Sj1==l);
      if (sumSj1== 0){
        res(l) = R::rnorm(0, 1) * sqrt(u2_alpha) + a_xi_alpha;
      } else {
        double sumSj1Sj20 = sum((Sj1==l) % (Sj2 ==0));
        double sumSj1Sj21 = sum((Sj1==l) % (Sj2 ==1));
        double sci2 = sumSj1Sj21 + sumSj1Sj20 * pow(w_l_a(l)/(1-w_l_a(l)), 2);
        arma::uvec idx1 = find( (Sj1==l)  % (Sj2==1) );
        arma::uvec idx0 = find( (Sj1==l)  % (Sj2==0) );
        double p_var = 1.0/(1.0/u2_alpha+ n * sci2/sig2);
        double tr = sum(sum(RSS.cols(idx1))) - w_l_a(l)/(1-w_l_a(l)) * sum(sum(RSS.cols(idx0) - nu_alpha/(1-w_l_a(l)) ));
        double p_m = (a_xi_alpha/u2_alpha + tr/sig2) * p_var;
        res(l) = R::rnorm(p_m, sqrt(p_var));

      }
  }
  return(res);
}


















