from pathlib import Path
import pandas as pd
import numpy as np
import pymc as pm


ROOT = Path(__file__).parent.parent.resolve()
DATA = ROOT / 'data'


if __name__ == '__main__':

	n_chains = 10
	generations = np.arange(1, 10)

	df = pd.read_csv(DATA / 'exp3.csv')

	df_lrn = df[  (df['condition'].str.contains('^dif_lrn') & (df['generation'] > 0))  ]
	cst_matrix_lrn = np.array(df_lrn['cost']).reshape((n_chains, -1)).T
	gen_matrix_lrn = np.repeat([generations], n_chains, axis=0).T
	gen_matrix_lrn -= 5 # center generation (gen 5 becomes intercept)

	df_com = df[  (df['condition'].str.contains('^dif_com') & (df['generation'] > 0))  ]
	cst_matrix_com = np.array(df_com['cost']).reshape((n_chains, -1)).T
	gen_matrix_com = np.repeat([generations], n_chains, axis=0).T
	gen_matrix_com -= 5 # center generation (gen 5 becomes intercept)

	cst_matrix = np.stack((cst_matrix_lrn, cst_matrix_com), axis=2)
	gen_matrix = np.stack((gen_matrix_lrn, gen_matrix_com), axis=2)

	coords = {
		'condition': ['lrn', 'com'],
		'chain_id': list(range(n_chains)),
		'generation': generations,
	}

	with pm.Model(coords=coords) as model:

		α_m = pm.Normal('α_m', mu=0, sigma=2, dims='condition')
		b1_m = pm.Normal('b1_m', mu=0, sigma=2, dims='condition')
		b2_m = pm.Normal('b2_m', mu=0, sigma=2, dims='condition')
		b3_m = pm.Normal('b3_m', mu=0, sigma=2, dims='condition')

		α_sd = pm.Exponential('α_sd', lam=1, dims='condition')
		b1_sd = pm.Exponential('b1_sd', lam=1, dims='condition')
		b2_sd = pm.Exponential('b2_sd', lam=1, dims='condition')
		b3_sd = pm.Exponential('b3_sd', lam=1, dims='condition')

		α = pm.Normal('α', mu=α_m, sigma=α_sd, dims=('chain_id', 'condition'))
		b1 = pm.Normal('b1', mu=b1_m, sigma=b1_sd, dims=('chain_id', 'condition'))
		b2 = pm.Normal('b2', mu=b2_m, sigma=b2_sd, dims=('chain_id', 'condition'))
		b3 = pm.Normal('b3', mu=b3_m, sigma=b3_sd, dims=('chain_id', 'condition'))

		s = pm.Exponential('s', lam=1, dims='condition')

		mu = α + b1 * gen_matrix + b2 * gen_matrix ** 2 + b3 * gen_matrix ** 3

		pm.Normal('cost', mu=mu, sigma=s, observed=cst_matrix, dims=('generation', 'chain_id', 'condition'))

		pm.Deterministic('pred_lrn', α_m[0] + b1_m[0] * gen_matrix[:, 0, 0] + b2_m[0] * gen_matrix[:, 0, 0] ** 2 + b3_m[0] * gen_matrix[:, 0, 0] ** 3)
		pm.Deterministic('pred_com', α_m[1] + b1_m[1] * gen_matrix[:, 0, 0] + b2_m[1] * gen_matrix[:, 0, 0] ** 2 + b3_m[1] * gen_matrix[:, 0, 0] ** 3)
		pm.Deterministic('α_diff', α_m[0] - α_m[1])

		trace = pm.sample(1000, tune=2000, target_accept=0.9)

		trace.to_netcdf(DATA / 'exp1_cube.netcdf')
