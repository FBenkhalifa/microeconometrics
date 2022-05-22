import textwrap
import stan


_STAN_CODE_: str = textwrap.dedent(
    """
    data {
      int<lower=0> N;         // number of observations
      int<lower=0> R;         // number of outcome classes
      int<lower=0> K;         // number of covariates
      matrix[N, K] x;         // The covariates
      int y[N, R];       // the outcome
    }
    parameters {
      vector[K] beta;         // estimated macroeconomic effects before corona
      vector[R] offsets;         // estimated macroeconomic effects before corona
    }
    transformed parameters {
      matrix[N, R] slope = rep_matrix(x * beta, R);
      matrix[N, R] offset_matrix = rep_matrix(offsets', N);
      matrix[N, R] class_probs = offset_matrix + slope;
      // Normalize class probabilities
        for (r in 1:R) {
            class_probs[, r] = softmax(class_probs[, r]);
        }
    }
    model {
      beta ~ normal(0, 1);
      offsets ~ normal(1, 1);

      for (r in 1:R) {
        y[:, r] ~ categorical(class_probs[:, r]);
      }
    }
    generated quantities{
    }
    """
)

schools_data = {
    "N": 5,
    "R": 3,
    "K": 5,
    "x": [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
    ],
    "y": [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ],
}

posterior = stan.build(_STAN_CODE_, data=schools_data)

fit = posterior.sample(num_chains=4, num_samples=1000)
