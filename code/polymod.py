import pandas as pd
import numpy as np
import pymc as pm
import arviz as az


def plot_predictive(axis, trace, var):
	generation = np.arange(1, 10)
	az.plot_hdi(generation, trace.posterior[var], hdi_prob=0.95, smooth=False, color='gray', ax=axis)
	axis.plot(generation, trace.posterior[var].mean(('chain', 'draw')), color='black')
	axis.set_ylim(0, 2)


if __name__ == '__main__':

	df = pd.read_csv('data/exp3.csv')

	df_lrn = df[  (df['condition'].str.contains('^con_lrn'))  ]
	cst_matrix_lrn = np.array(df_lrn['cost']).reshape((10, 10)).T
	gen_matrix_lrn = np.repeat([range(10)], 10, axis=0).T
	gen_matrix_lrn -= 5 # center generation

	df_com = df[  (df['condition'].str.contains('^con_com'))  ]
	cst_matrix_com = np.array(df_com['cost']).reshape((10, 10)).T
	gen_matrix_com = np.repeat([range(10)], 10, axis=0).T
	gen_matrix_com -= 5 # center generation

	cst_matrix = np.stack((cst_matrix_lrn, cst_matrix_com), axis=2) + 0.0001
	gen_matrix = np.stack((gen_matrix_lrn, gen_matrix_com), axis=2)

	coords = {
		'condition': ['lrn', 'com'],
		'chain_id': list(range(10)),
		'generation': list(range(10))
	}

	with pm.Model(coords=coords) as model:

		b0_m = pm.Normal('b0_m', mu=0.1, sigma=0.5, dims='condition')
		b1_m = pm.Normal('b1_m', mu=0.1, sigma=0.5, dims='condition')
		# b2_m = pm.Normal('b2_m', mu=0.1, sigma=0.5, dims='condition')
		# b3_m = pm.Normal('b3_m', mu=0.1, sigma=0.5, dims='condition')

		b0_sd = pm.Exponential('b0_sd', lam=1, dims='condition')
		b1_sd = pm.Exponential('b1_sd', lam=1, dims='condition')
		# b2_sd = pm.Exponential('b2_sd', lam=1, dims='condition')
		# b3_sd = pm.Exponential('b3_sd', lam=1, dims='condition')

		b0 = pm.Normal('b0', mu=b0_m, sigma=b0_sd, dims=('chain_id', 'condition'))
		b1 = pm.Normal('b1', mu=b1_m, sigma=b1_sd, dims=('chain_id', 'condition'))
		# b2 = pm.Normal('b2', mu=b2_m, sigma=b2_sd, dims=('chain_id', 'condition'))
		# b3 = pm.Normal('b3', mu=b3_m, sigma=b3_sd, dims=('chain_id', 'condition'))

		s = pm.Exponential('s', lam=1, dims='condition')

		m = b0 + b1 * gen_matrix# + b2 * gen_matrix ** 2 + b3 * gen_matrix ** 3

		mu = pm.Deterministic('mu', pm.math.exp(m))

		pm.Gamma('cost', mu=mu, sigma=s, observed=cst_matrix, dims=('generation', 'chain_id', 'condition'))

		pm.Deterministic('pred_lrn', pm.math.exp(b0_m[0] + b1_m[0] * gen_matrix[:, 0, 0]))# + b2_m[0] * gen_matrix[:, 0, 0] ** 2 + b3_m[0] * gen_matrix[:, 0, 0] ** 3))
		pm.Deterministic('pred_com', pm.math.exp(b0_m[1] + b1_m[1] * gen_matrix[:, 0, 0]))# + b2_m[1] * gen_matrix[:, 0, 0] ** 2 + b3_m[1] * gen_matrix[:, 0, 0] ** 3))

		trace = pm.sample(1000, tune=2000)

		trace.to_netcdf('data/exp3_con.netcdf')
