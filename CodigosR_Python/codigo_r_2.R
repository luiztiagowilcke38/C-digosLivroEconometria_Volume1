library(nloptr)
library(Rcpp)

# Usando C++ In-line para compilar o nucleo computacional pesado
cppFunction('
  double exp_sum_cpp(NumericVector beta, NumericMatrix X, NumericVector Y) {
    int n = X.nrow();
    int k = X.ncol();
    double ll = 0;
    
    for(int i = 0; i < n; ++i) {
      double eta = 0;
      for(int j = 0; j < k; ++j) {
        eta += X(i,j) * beta[j];
      }
      double pi = 1.0 / (1.0 + exp(-eta));
      // Clamp para num stability
      if(pi < 1e-15) pi = 1e-15;
      if(pi > 1 - 1e-15) pi = 1 - 1e-15;
      
      ll += Y[i]*log(pi) + (1-Y[i])*log(1-pi);
    }
    return -ll; // neg-log-likelihood
  }
')

# Funcao de Avaliacao Roteada em R
estimar_logit_nlopt <- function(y, X_mat) {
  
  obj_function <- function(b) {
    return(exp_sum_cpp(b, X_mat, y))
  }
  
  # ALGORITMO: L-BFGS (Low-Storage BFGS) - sem gradiente analitico explicito (usa dif fnita local pelo solver mas num kernel subjacente em C++)
  opt_res <- nloptr(
    x0 = rep(0, ncol(X_mat)),
    eval_f = obj_function,
    lb = rep(-10, ncol(X_mat)), # Restricoes de caixa para estabilidade
    ub = rep(10, ncol(X_mat)),
    opts = list(''algorithm'' = ''NLOPT_LN_LBFGS'', # Derivada numerica interna por NLOpt
                ''xtol_rel'' = 1.0e-9,
                ''maxeval'' = 1000)
  )
  
  # Matriz Jacobiana e Hessiana Inferida
  library(numDeriv)
  hess <- hessian(obj_function, opt_res$solution)
  vcov <- solve(hess)
  se <- sqrt(diag(vcov))
  
  return(list(coef = opt_res$solution, se = se))
}
