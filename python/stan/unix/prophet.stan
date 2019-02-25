functions {
  matrix get_changepoint_matrix(vector t, vector t_change, int T, int S) {
    // Assumes t and t_change are sorted.
    matrix[T, S] A;
    row_vector[S] a_row;
    int cp_idx;

    // Start with an empty matrix.
    A = rep_matrix(0, T, S);
    a_row = rep_row_vector(0, S);
    cp_idx = 1;

    // Fill in each row of A.
    for (i in 1:T) {
      while ((cp_idx <= S) && (t[i] >= t_change[cp_idx])) {
        a_row[cp_idx] = 1;
        cp_idx = cp_idx + 1;
      }
      A[i] = a_row;
    }
    return A;
  }

  // Logistic trend functions

  matrix logistic_gamma(
    vector k,
    vector m,
    matrix delta,
    vector t_change,
    int S,
    int N
  ) {
    matrix[S,N] gamma;  // adjusted offsets, for piecewise continuity
    // matrix[S+1,N] k_s;  // actual rate in each segment
    real k_pr;
    real m_pr;

    // Compute the rate in each segment
    // for (j in 1:N) {
    //  k_s[1,j] = k[j]
    //  for (i in 1:S) {
    //    k_s[i+1,j] = k_s[i,j] + delta[i,j]
    //  }
    // }
    //
    // Piecewise offsets
    // for (j in 1:N) {
    //   m_pr = m[j]; // The offset in the previous segment
    //   for (i in 1:S) {
    //     gamma[i,j] = (t_change[i] - m_pr) * (1 - k_s[i,j] / k_s[i+1,j]);
    //     m_pr = m_pr + gamma[i,j];  // update for the next segment
    //   }
    // }

    // Piecewise offsets
    for (j in 1:N) {
      k_pr = k[j]; // The rate in the previous segment
      m_pr = m[j]; // The offset in the previous segment
      for (i in 1:S) {
        gamma[i,j] = (t_change[i] - m_pr) * (1 - k_pr / (k_pr + delta[i,j]));
        k_pr = k_pr + delta[i,j];  // update for the next segment
        m_pr = m_pr + gamma[i,j];  // update for the next segment
      }
    }
    return gamma;
  }

  matrix logistic_trend(
    vector k,
    vector m,
    matrix delta,
    vector t,
    vector cap,
    matrix A,
    vector t_change,
    int T,
    int S,
    int N
  ) {
    matrix[S,N] gamma;

    gamma = logistic_gamma(k, m, delta, t_change, S, N);
    return rep_matrix(cap, N) .* inv_logit(
      (rep_matrix(k', T) + A * delta) .* (rep_matrix(t, N) - (rep_matrix(m', T) + A * gamma))
    );
  }

  // Linear trend function

  matrix linear_trend(
    vector k,
    vector m,
    matrix delta,
    vector t,
    matrix A,
    vector t_change,
    int T,
    int S,
    int N
  ) {
    return (rep_matrix(k', T) + A * delta) .* rep_matrix(t, N)
            + (rep_matrix(m', T) + A * (-rep_matrix(t_change, N) .* delta));
  }
}

data {
  int T;                // Number of time periods
  int<lower=1> N;       // Number of independent variables
  int<lower=1> K;       // Number of regressors
  vector[T] t;          // Time
  vector[T] cap;        // Capacities for logistic trend
  vector[T] y;          // Time series for dependent variable
  matrix[T,N] X;        // Time series for independent variables
  int S;                // Number of changepoints
  vector[S] t_change;   // Times of trend changepoints
  matrix[T,K] Z;        // Regressors
  vector[K] sigmas;     // Scale on seasonality prior
  real<lower=0> tau;    // Scale on changepoints prior
  int trend_indicator;  // 0 for linear, 1 for logistic
  vector[K] s_a;        // Indicator of additive features
  vector[K] s_m;        // Indicator of multiplicative features
}

transformed data {
  matrix[T, S] A;
  A = get_changepoint_matrix(t, t_change, T, S);
}

parameters {
  vector[N] k;              // Base trend growth rate
  vector[N] m;              // Trend offset
  real<lower=0> sigma_obs;  // Observation noise
  matrix[S,N] delta;        // Trend rate adjustments
  matrix[K,N] beta;         // Regressor coefficients
}

model {
  //priors
  k ~ normal(0, 5);
  m ~ normal(0, 5);
  sigma_obs ~ normal(0, 0.5);
  for (j in 1:N) {
    delta[:,j] ~ double_exponential(0, tau);
    beta[:,j] ~ normal(0, sigmas);
  }

  // Likelihood
  if (trend_indicator == 0) {
    y ~ normal(
      rows_dot_product((
        linear_trend(k, m, delta, t, A, t_change, T, S, N)
        .* (1 + Z * (beta .* rep_matrix(s_m, N)))
        + Z * (beta .* rep_matrix(s_a, N))
      ), X), sigma_obs
    );
  } else if (trend_indicator == 1) {
    y ~ normal(
      rows_dot_product((
        logistic_trend(k, m, delta, t, cap, A, t_change, T, S, N)
        .* (1 + Z * (beta .* rep_matrix(s_m, N)))
        + Z * (beta .* rep_matrix(s_a, N))
      ), X), sigma_obs
    );
  }
}
