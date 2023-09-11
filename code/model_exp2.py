from pathlib import Path
import pandas as pd
import numpy as np
import pymc as pm


ROOT = Path(__file__).parent.parent.resolve()
DATA = ROOT / 'data'


# if __name__ == '__main__':

# 	chains = list(range(0, 10))
# 	generations = list(range(1, 10))

# 	df = pd.read_csv(DATA / 'exp3.csv')

# 	df_lrn = df[  (df['condition'].str.contains('^con_lrn') & (df['generation'] >= min(generations)))  ]
# 	cost_lrn = np.array(df_lrn['cost']).reshape((len(chains), len(generations))).T

# 	df_com = df[  (df['condition'].str.contains('^con_com') & (df['generation'] >= min(generations)))  ]
# 	cost_com = np.array(df_com['cost']).reshape((len(chains), len(generations))).T

# 	cost = np.stack((cost_lrn, cost_com), axis=2)

# 	e1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# 	e1 = np.repeat([e1], len(chains), axis=0).T
# 	e1 = np.stack((e1, e1), axis=2)
# 	e1 -= 5

# 	e2 = np.array([0, 0, 0, -1, 0, 1, 2, 3, 4])
# 	e2 = np.repeat([e2], len(chains), axis=0).T
# 	e2 = np.stack((e2, e2), axis=2)

# 	e3 = np.array([0, 0, 0, 0, 0, 0, 2, 3, 4])
# 	e3 = np.repeat([e3], len(chains), axis=0).T
# 	e3 = np.stack((e3, e3), axis=2)

# 	coords = {
# 		'condition': ['lrn', 'com'],
# 		'chain_id': chains,
# 		'generation': generations,
# 	}

# 	with pm.Model(coords=coords) as model:

# 		# Hyperpriors

# 		α_m = pm.Normal('α_m', mu=0, sigma=1, dims='condition')
# 		β1_m = pm.Normal('β1_m', mu=0, sigma=1, dims='condition')
# 		β2_m = pm.Normal('β2_m', mu=0, sigma=1, dims='condition')
# 		β3_m = pm.Normal('β3_m', mu=0, sigma=1, dims='condition')

# 		α_sd = pm.Exponential('α_sd', lam=1, dims='condition')
# 		β1_sd = pm.Exponential('β1_sd', lam=1, dims='condition')
# 		β2_sd = pm.Exponential('β2_sd', lam=1, dims='condition')
# 		β3_sd = pm.Exponential('β3_sd', lam=1, dims='condition')

# 		# Priors

# 		α = pm.Normal('α', mu=α_m, sigma=α_sd, dims=('chain_id', 'condition'))
# 		β1 = pm.Normal('β1', mu=β1_m, sigma=β1_sd, dims=('chain_id', 'condition'))
# 		β2 = pm.Normal('β2', mu=β2_m, sigma=β2_sd, dims=('chain_id', 'condition'))
# 		β3 = pm.Normal('β3', mu=β3_m, sigma=β3_sd, dims=('chain_id', 'condition'))
# 		σ = pm.Exponential('σ', lam=1, dims='condition')

# 		# Model

# 		μ = α + β1*e1 + β2*e2 + β3*e3

# 		# Likelihood

# 		pm.Normal('cost', mu=μ, sigma=σ, observed=cost, dims=('generation', 'chain_id', 'condition'))

# 		# Deterministic parameters

# 		pm.Deterministic('diff_α', α_m[1] - α_m[0])
# 		pm.Deterministic('diff_β1', β1_m[1] - β1_m[0])
# 		pm.Deterministic('diff_β2', β2_m[1] - β2_m[0])
# 		pm.Deterministic('diff_β3', β3_m[1] - β3_m[0])
# 		pm.Deterministic('diff_σ', σ[1] - σ[0])

# 		pm.Deterministic('pred_lrn', α_m[0] + β1_m[0] * e1[:,0,0] + β2_m[0] * e2[:,0,0] + β3_m[0] * e3[:,0,0])
# 		pm.Deterministic('pred_com', α_m[1] + β1_m[1] * e1[:,0,0] + β2_m[1] * e2[:,0,0] + β3_m[1] * e3[:,0,0])

# 		# Sampling

# 		trace = pm.sample(10000, tune=2000, target_accept=0.99, chains=6, cores=6)
# 		trace.to_netcdf(DATA / 'exp2.netcdf')

if __name__ == '__main__':

	chains = list(range(0, 10))
	generations = list(range(1, 10))

	df = pd.read_csv(DATA / 'exp3.csv')

	df_lrn = df[  (df['condition'].str.contains('^con_lrn') & (df['generation'] >= min(generations)))  ]
	cost_lrn = np.array(df_lrn['cost']).reshape((len(chains), len(generations))).T

	df_com = df[  (df['condition'].str.contains('^con_com') & (df['generation'] >= min(generations)))  ]
	cost_com = np.array(df_com['cost']).reshape((len(chains), len(generations))).T

	cost = np.stack((cost_lrn, cost_com), axis=2)

	c = np.zeros((3, 10, 3, 2))
	g = np.zeros((3, 10, 3, 2))
	for (gn, ch, ep, cd) in np.ndindex(c.shape):
		gen_index = ep * 3 + gn
		c[gn, ch, ep, cd] = cost[gen_index, ch, cd]
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

		pm.Normal('c', mu=μ, sigma=σ, observed=c, dims=('generation', 'chain_id', 'epoch', 'condition'))

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

		trace = pm.sample(10000, tune=2000, target_accept=0.99, chains=6, cores=6)
		trace.to_netcdf(DATA / 'exp2.netcdf')
