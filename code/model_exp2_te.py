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

	df_lrn = df[  (df['condition'].str.contains('^con_lrn') & (df['generation'] >= min(generations)))  ]
	error_lrn = np.array(df_lrn['error']).reshape((len(chains), len(generations))).T

	df_com = df[  (df['condition'].str.contains('^con_com') & (df['generation'] >= min(generations)))  ]
	error_com = np.array(df_com['error']).reshape((len(chains), len(generations))).T

	error = np.stack((error_lrn, error_com), axis=2)

	e = np.zeros((3, 10, 3, 2))
	g = np.zeros((3, 10, 3, 2))
	for (gn, ch, ep, cd) in np.ndindex(e.shape):
		gen_index = ep * 3 + gn
		e[gn, ch, ep, cd] = error[gen_index, ch, cd]
		g[gn, ch, ep, cd] = (gen_index % 3) - 1

	coords = {
		'condition': ['lrn', 'com'],
		'epoch': [1, 2, 3],
		'chain_id': chains,
		'generation': [1, 2, 3],
	}

	with pm.Model(coords=coords) as model:

		# Hyperpriors

		α_m = pm.Normal('α_m', mu=0, sigma=1, dims=('epoch', 'condition'))
		β_m = pm.Normal('β_m', mu=0, sigma=1, dims=('epoch', 'condition'))

		α_sd = pm.Exponential('α_sd', lam=1, dims=('epoch', 'condition'))
		β_sd = pm.Exponential('β_sd', lam=1, dims=('epoch', 'condition'))

		# Priors

		α = pm.Normal('α', mu=α_m, sigma=α_sd, dims=('chain_id', 'epoch', 'condition'))
		β = pm.Normal('β', mu=β_m, sigma=β_sd, dims=('chain_id', 'epoch', 'condition'))
		σ = pm.Exponential('σ', lam=1, dims='condition')

		# Model

		μ = α + β * g

		# Likelihood

		pm.Normal('e', mu=μ, sigma=σ, observed=e, dims=('generation', 'chain_id', 'epoch', 'condition'))

		# Deterministic parameters

		pm.Deterministic('diff_α1', α_m[0,1] - α_m[0,0])
		pm.Deterministic('diff_α2', α_m[1,1] - α_m[1,0])
		pm.Deterministic('diff_α3', α_m[2,1] - α_m[2,0])
		pm.Deterministic('diff_β1', β_m[0,1] - β_m[0,0])
		pm.Deterministic('diff_β2', β_m[1,1] - β_m[1,0])
		pm.Deterministic('diff_β3', β_m[2,1] - β_m[2,0])
		pm.Deterministic('diff_σ', σ[1] - σ[0])

		for epoch in range(3):
			for gen in range(3):
				actual_gen = epoch * 3 + gen + 1
				pm.Deterministic(f'pred_lrn_{actual_gen}', α_m[epoch,0] + β_m[epoch,0] * (gen - 1))
				pm.Deterministic(f'pred_com_{actual_gen}', α_m[epoch,1] + β_m[epoch,1] * (gen - 1))

		# Sampling

		trace = pm.sample(2000, tune=2000, target_accept=0.95, chains=4, cores=4)
		trace.to_netcdf(DATA / 'exp2_error.netcdf')
