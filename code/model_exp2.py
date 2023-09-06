from pathlib import Path
import pandas as pd
import numpy as np
import pymc as pm


ROOT = Path(__file__).parent.parent.resolve()
DATA = ROOT / 'data'


if __name__ == '__main__':

	n_chains = 10
	# generations = np.arange(1, 10)

	e1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	e2 = np.array([0, 0, 0, 0, 1, 2, 3, 4, 5, 6])
	e3 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 2, 3])

	df = pd.read_csv(DATA / 'exp3.csv')

	df_lrn = df[  (df['condition'].str.contains('^con_lrn') & (df['generation'] > -1))  ]
	cst_lrn = np.array(df_lrn['cost']).reshape((n_chains, -1)).T

	df_com = df[  (df['condition'].str.contains('^con_com') & (df['generation'] > -1))  ]
	cst_com = np.array(df_com['cost']).reshape((n_chains, -1)).T

	cst = np.stack((cst_lrn, cst_com), axis=2)

	e1 = np.repeat([e1], n_chains, axis=0).T
	e1 = np.stack((e1, e1), axis=2)

	e2 = np.repeat([e2], n_chains, axis=0).T
	e2 = np.stack((e2, e2), axis=2)

	e3 = np.repeat([e3], n_chains, axis=0).T
	e3 = np.stack((e3, e3), axis=2)

	coords = {
		'condition': ['lrn', 'com'],
		'chain_id': list(range(n_chains)),
		'generation': list(range(10)),
	}

	with pm.Model(coords=coords) as model:

		α_m = pm.Normal('α_m', mu=0, sigma=1, dims='condition')
		β1_m = pm.Normal('β1_m', mu=0, sigma=1, dims='condition')
		β2_m = pm.Normal('β2_m', mu=0, sigma=1, dims='condition')
		β3_m = pm.Normal('β3_m', mu=0, sigma=1, dims='condition')

		α_sd = pm.Exponential('α_sd', lam=1, dims='condition')
		β1_sd = pm.Exponential('β1_sd', lam=1, dims='condition')
		β2_sd = pm.Exponential('β2_sd', lam=1, dims='condition')
		β3_sd = pm.Exponential('β3_sd', lam=1, dims='condition')

		α = pm.Normal('α', mu=α_m, sigma=α_sd, dims=('chain_id', 'condition'))
		β1 = pm.Normal('β1', mu=β1_m, sigma=β1_sd, dims=('chain_id', 'condition'))
		β2 = pm.Normal('β2', mu=β2_m, sigma=β2_sd, dims=('chain_id', 'condition'))
		β3 = pm.Normal('β3', mu=β3_m, sigma=β3_sd, dims=('chain_id', 'condition'))

		s = pm.Exponential('s', lam=1, dims='condition')

		mu = α + β1*e1 + β2*e2 + β3*e3

		pm.Normal('cost', mu=mu, sigma=s, observed=cst, dims=('generation', 'chain_id', 'condition'))

		pm.Deterministic('pred_lrn', α_m[0] + β1_m[0] * e1[:, 0, 0] + β2_m[0] * e2[:, 0, 0] + β3_m[0] * e3[:, 0, 0])
		pm.Deterministic('pred_com', α_m[1] + β1_m[1] * e1[:, 0, 0] + β2_m[1] * e2[:, 0, 0] + β3_m[1] * e3[:, 0, 0])

		trace = pm.sample(2000, tune=2000, target_accept=0.99, chains=6, cores=6)

		trace.to_netcdf(DATA / 'exp2.netcdf')
