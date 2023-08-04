import numpy as np
import pandas as pd
import bambi as bmb
import arviz as az


def center_variable(df, var):
	width = max(df[var]) - min(df[var])
	midpoint = min(df[var]) + width / 2
	df[var] = df[var] - midpoint
	return df

def generate_random_dataset(n_chains, n_generations, beta, noise_sd):
	gen = np.array(beta * np.arange(1, n_generations + 1))
	chains = np.array([gen + np.random.normal(0, noise_sd, n_generations) for _ in range(n_chains)])
	chains = np.clip(chains, 0, 3.16)
	return chains

def plot_dataset(data1, data2):
	fig, axes = az.utils.plt.subplots(1, 2)
	for chain in data1:
		axes[0].plot(range(1, 10), chain)
	for chain in data2:
		axes[1].plot(range(1, 10), chain)
	axes[0].set_ylim(0, 3)
	axes[1].set_ylim(0, 3)
	az.utils.plt.show()


if __name__ == '__main__':

	np.random.seed(117)

	n_chains = 10
	n_generations = 9

	data1 = generate_random_dataset(n_chains, n_generations, 0.3, 0.3)
	data2 = generate_random_dataset(n_chains, n_generations, 0.25, 0.3)

	# plot_dataset(data1, data2)
	# quit()

	chain_ids = []
	np.repeat

	df = pd.DataFrame({
		'cost': list(data1.flatten()) + list(data2.flatten()),
		'condition': ['lrn'] * (n_chains * n_generations) + ['com'] * (n_chains * n_generations),
		'generation': list(range(1, n_generations+1)) * (n_chains*2),
		'chain': np.repeat(range(n_chains*2), n_generations),
	})

	# df = pd.read_csv('../data/exp2.csv')
	# df = df[ df['generation'] > 0 ]
	df = center_variable(df, 'generation')
	df['chain_id'] = df[['condition', 'chain']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

	model = bmb.Model('cost ~ generation * condition + (1 + generation | chain_id)', df)
	results = model.fit(draws=1000, chains=4)
