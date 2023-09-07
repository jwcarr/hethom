from pathlib import Path
import pandas as pd
import numpy as np
import pymc as pm



ROOT = Path(__file__).parent.parent.resolve()
DATA = ROOT / 'data'


if __name__ == '__main__':

	chains = list(range(0, 10))
	generations = list(range(1, 10))

	df = pd.read_csv(DATA / 'exp3.csv')

	df_lrn = df[  (df['condition'].str.contains('^con_lrn') & (df['generation'] >= min(generations)))  ]
	cost_lrn = np.array(df_lrn['cost']).reshape((len(chains), len(generations))).T

	df_com = df[  (df['condition'].str.contains('^con_com') & (df['generation'] >= min(generations)))  ]
	cost_com = np.array(df_com['cost']).reshape((len(chains), len(generations))).T

	cost = np.stack((cost_lrn, cost_com), axis=2)

	e1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
	e1 = np.repeat([e1], len(chains), axis=0).T
	e1 = np.stack((e1, e1), axis=2)

	e2 = np.array([0, 0, 0, 1, 2, 3, 4, 5, 6])
	e2 = np.repeat([e2], len(chains), axis=0).T
	e2 = np.stack((e2, e2), axis=2)

	e3 = np.array([0, 0, 0, 0, 0, 0, 1, 2, 3])
	e3 = np.repeat([e3], len(chains), axis=0).T
	e3 = np.stack((e3, e3), axis=2)

	coords = {
		'condition': ['lrn', 'com'],
		'chain_id': chains,
		'generation': generations,
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

		σ = pm.Exponential('σ', lam=1, dims='condition')

		mu = α + β1*e1 + β2*e2 + β3*e3

		pm.Normal('cost', mu=mu, sigma=σ, observed=cost, dims=('generation', 'chain_id', 'condition'))

		pm.Deterministic('pred_lrn', α_m[0] + β1_m[0] * e1[:,0,0] + β2_m[0] * e2[:,0,0] + β3_m[0] * e3[:,0,0])
		pm.Deterministic('pred_com', α_m[1] + β1_m[1] * e1[:,0,0] + β2_m[1] * e2[:,0,0] + β3_m[1] * e3[:,0,0])

		trace = pm.sample(2000, tune=2000, target_accept=0.99, chains=4, cores=4)
		trace.to_netcdf(DATA / 'exp2.netcdf')


# if __name__ == '__main__':

# 	chains = list(range(0, 10))
# 	generations = list(range(1, 10))

# 	df = pd.read_csv(DATA / 'exp3.csv')

# 	df_lrn = df[  (df['condition'].str.contains('^con_lrn') & (df['generation'] >= min(generations)))  ]
# 	cost_lrn = np.array(df_lrn['cost']).reshape((len(chains), len(generations))).T

# 	df_com = df[  (df['condition'].str.contains('^con_com') & (df['generation'] >= min(generations)))  ]
# 	cost_com = np.array(df_com['cost']).reshape((len(chains), len(generations))).T

# 	cost = np.stack((cost_lrn, cost_com), axis=2)

# 	generation = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# 	generation = np.repeat([generation], len(chains), axis=0).T
# 	generation = np.stack((generation, generation), axis=2)

# 	epoch = np.array([
# 		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
# 		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
# 		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
# 		[2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
# 		[2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
# 		[2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
# 	])
# 	epoch = np.stack((epoch, epoch), axis=2)

# 	coords = {
# 		'condition': ['lrn', 'com'],
# 		'chain_id': chains,
# 		'generation': generations,
# 		'epoch': [0, 1, 2],
# 	}

# 	with pm.Model(coords=coords) as model:

# 		α_m = pm.Normal('α_m', mu=0, sigma=1, dims='condition')
# 		β_m = pm.Normal('β_m', mu=0, sigma=1, dims=('epoch', 'condition'))

# 		α_sd = pm.Exponential('α_sd', lam=1, dims='condition')
# 		β_sd = pm.Exponential('β_sd', lam=1, dims=('epoch', 'condition'))

# 		α = pm.Normal('α', mu=α_m, sigma=α_sd, dims=('chain_id', 'condition'))
# 		β = pm.Normal('β', mu=β_m, sigma=β_sd, dims=('epoch', 'chain_id', 'condition'))

# 		s = pm.Exponential('s', lam=1, dims='condition')

# 		mu = α + β[epoch, :, :] * generation

# 		pm.Normal('cost', mu=mu, sigma=s, observed=cost, dims=('generation', 'chain_id', 'condition'))

# 		# pm.Deterministic('pred_lrn', α_m[0] + β_m[0] * e1[:,0,0] + β2_m[0] * e2[:,0,0] + β3_m[0] * e3[:,0,0])
# 		# pm.Deterministic('pred_com', α_m[1] + β_m[1] * e1[:,0,0] + β2_m[1] * e2[:,0,0] + β3_m[1] * e3[:,0,0])

# 		trace = pm.sample(1000, tune=1000, target_accept=0.8, chains=4, cores=4)

# 		trace.to_netcdf(DATA / 'exp2_reduced.netcdf')


# if __name__ == '__main__':
# from patsy import dmatrix

# 	chains = list(range(0, 10))
# 	generations = list(range(1, 10))

# 	df = pd.read_csv(DATA / 'exp3.csv')

# 	df_lrn = df[  (df['condition'].str.contains('^con_lrn') & (df['generation'] >= min(generations)))  ]
# 	cost = np.array(df_lrn['cost'])

# 	generation = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9] * 10)
# 	epoch = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2] * 10)

# 	knots = np.array([0.5, 3.5, 6.5, 9.5])
# 	B = dmatrix(
# 		'bs(generation, knots=knots, degree=1, include_intercept=True) - 1',
# 		{'generation': generation, 'knots': knots[1:-1]}
# 	)

# 	coords = {
# 		'splines': np.arange(B.shape[1]),
# 	}

# 	with pm.Model(coords=coords) as model:

# 		α = pm.Normal('α', mu=0, sigma=1)
# 		β = pm.Normal('β', mu=0, sigma=1, dims='splines')
# 		s = pm.Exponential('s', lam=1)

# 		mu = pm.Deterministic('mu', α + pm.math.dot(np.asarray(B, order='F'), β.T))

# 		pm.Normal('cost', mu=mu, sigma=s, observed=cost)

# 		trace = pm.sample(1000, tune=1000, target_accept=0.8, chains=4, cores=4)
# 		trace.to_netcdf(DATA / 'exp2_reduced.netcdf')


# if __name__ == '__main__':

# 	chains = list(range(0, 10))
# 	generations = list(range(1, 10))

# 	df = pd.read_csv(DATA / 'exp3.csv')

# 	df_lrn = df[  (df['condition'].str.contains('^con_lrn') & (df['generation'] >= min(generations)))  ]
# 	cost = np.array(df_lrn['cost'])

# 	generation = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9] * 10)
# 	epoch = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2] * 10)

# 	coords = {
# 		'epoch': [0, 1, 2]
# 	}

# 	with pm.Model(coords=coords) as model:

# 		α = pm.Normal('α', mu=0, sigma=1)
# 		β = pm.Normal('β', mu=0, sigma=1, dims='epoch')
# 		s = pm.Exponential('s', lam=1)

# 		mu = pm.Deterministic('mu', α + β[epoch] * generation)

# 		pm.Normal('cost', mu=mu, sigma=s, observed=cost)

# 		trace = pm.sample(1000, tune=1000, target_accept=0.8, chains=4, cores=4)
# 		trace.to_netcdf(DATA / 'exp2_reduced.netcdf')

