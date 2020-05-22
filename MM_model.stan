functions{

  // Theoretical MM enzyme equation
  vector MM(real k_cat, real K_M, vector conc){
    return  k_cat * (conc ./ (K_M + conc));
  }

  // Observed rate, units of mAU per second
  vector a_t(real rate, vector t, real a0){
    return rate * t + a0;
  }
  
}


data{

  // Total number of data points
  int N;

  // Number of rate measurements
  int M;
  
  // Number of background rate (no substrate) measurements
  // assumed to be the first 'M0' rates in the array
  int M0;
    
  // The data
  vector[N] t;
  matrix[M,N] a;
  
  // Known constants
  vector[M] conc; // uM
  real epsilon; // mM/s
  real c_Enz; // nM
  
  // Scaling factor
  int scaling_factor;
  
  // For PPCs
  int max_conc;
  vector[max_conc+1] conc_ppc;
    
}


parameters{

  real<lower=0> V_max;

  real<lower=0> K_M;

  real<lower=0> sigma_k;
  vector<lower=0>[M] sigma_a;

  vector[M] rate;
  vector[M] a0;
  real background;

}


transformed parameters{
    
  // In units of uM/s
  vector[M] v0 = (rate/scaling_factor)/epsilon;
  
  // In units of /s
  vector[M] k = v0/(c_Enz/1000);
  real k_cat = V_max/(c_Enz/1000);

}


model{
  
  // Priors
  k_cat ~ lognormal(log(150), 2.5); // per second
  K_M ~ lognormal(log(500), 1.5); // uM
  
  background ~ std_normal();
  a0 ~ std_normal();

  sigma_k ~ std_normal();
  sigma_a ~ std_normal();

  // Likelihood
  for (m in 1:M0){
    a[m] ~ normal(a_t(background, t, a0[m]), sigma_a[m]);
  }
  
  for (m in 1:M){
    a[m] ~ normal(a_t(rate[m] + background, t, a0[m]), sigma_a[m]);
  }
  
  k ~ normal(MM(k_cat, K_M, conc), sigma_k);
  
}


generated quantities{
  
  real k_ppc[max_conc+1];
  
  // Draw posterior predictive data set
  k_ppc = normal_rng(MM(k_cat, K_M, conc_ppc), sigma_k);

}