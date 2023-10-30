from pathlib import Path
import pandas as pd
import numpy as np
import pymc as pm


ROOT = Path(__file__).parent.parent.resolve()
DATA = ROOT / 'data'


if __name__ == '__main__':

	chains = list(range(0, 10))
	generations = list(range(1, 10))

	df = pd.read_csv(DATA / 'exp.csv')

	df_lrn = df[  (df['condition'].str.contains('^dif_lrn') & (df['generation'] >= min(generations)))  ]
	cost_lrn = np.array(df_lrn['cost']).reshape((len(chains), len(generations))).T

	df_com = df[  (df['condition'].str.contains('^dif_com') & (df['generation'] >= min(generations)))  ]
	cost_com = np.array(df_com['cost']).reshape((len(chains), len(generations))).T

	c = np.stack((cost_lrn, cost_com), axis=2)
	
	g = np.repeat([generations], len(chains), axis=0).T
	g = np.stack((g, g), axis=2)
	g -= 5 # center generation

	coords = {
		'condition': ['lrn', 'com'],
		'chain_id': chains,
		'generation': generations,
	}

	with pm.Model(coords=coords) as model:

		# Hyperpriors

		α_m = pm.Normal('α_m', mu=1, sigma=1, dims='condition')
		β1_m = pm.Normal('β1_m', mu=0, sigma=1, dims='condition')
		β2_m = pm.Normal('β2_m', mu=0, sigma=1, dims='condition')
		β3_m = pm.Normal('β3_m', mu=0, sigma=1, dims='condition')

		α_sd = pm.Exponential('α_sd', lam=1, dims='condition')
		β1_sd = pm.Exponential('β1_sd', lam=1, dims='condition')
		β2_sd = pm.Exponential('β2_sd', lam=1, dims='condition')
		β3_sd = pm.Exponential('β3_sd', lam=1, dims='condition')

		# Priors

		α = pm.Normal('α', mu=α_m, sigma=α_sd, dims=('chain_id', 'condition'))
		β1 = pm.Normal('β1', mu=β1_m, sigma=β1_sd, dims=('chain_id', 'condition'))
		β2 = pm.Normal('β2', mu=β2_m, sigma=β2_sd, dims=('chain_id', 'condition'))
		β3 = pm.Normal('β3', mu=β3_m, sigma=β3_sd, dims=('chain_id', 'condition'))
		σ = pm.Exponential('σ', lam=1, dims='condition')

		# Model

		μ = α + β1*g + β2*g**2 + β3*g**3

		# Likelihood

		pm.Normal('c', mu=μ, sigma=σ, observed=c, dims=('generation', 'chain_id', 'condition'))

		# Deterministic parameters

		pm.Deterministic('diff_α', α_m[1] - α_m[0])
		pm.Deterministic('diff_β1', β1_m[1] - β1_m[0])
		pm.Deterministic('diff_β2', β2_m[1] - β2_m[0])
		pm.Deterministic('diff_β3', β3_m[1] - β3_m[0])
		pm.Deterministic('diff_σ', σ[1] - σ[0])

		for gen in generations:
			centered_gen = gen - 5
			pm.Deterministic(f'pred_lrn_{gen}', α_m[0] + β1_m[0]*centered_gen + β2_m[0]*centered_gen**2 + β3_m[0]*centered_gen**3)
			pm.Deterministic(f'pred_com_{gen}', α_m[1] + β1_m[1]*centered_gen + β2_m[1]*centered_gen**2 + β3_m[1]*centered_gen**3)

		# Sampling

		trace = pm.sample(10000, tune=2000, target_accept=0.99, chains=6, cores=6)
		trace.to_netcdf(DATA / 'exp1_cost.netcdf')
