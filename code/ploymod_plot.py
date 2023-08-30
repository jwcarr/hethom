import matplotlib.pyplot as plt
import arviz as az
import numpy as np
import pandas as pd


def plot_chains(axis, dataset, var, show_mean=False):
	chain_data = []
	for chain_i in sorted(dataset['chain'].unique()):
		chain_subset = dataset[ dataset['chain'] == chain_i ]
		chain_data.append(list(chain_subset[var]))
		axis.plot(chain_subset['generation'], chain_subset[var], label=f'Chain {chain_i + 1}')
	if show_mean:
		chain_data = np.array(chain_data)
		axis.plot(chain_subset['generation'], chain_data.mean(axis=0), label=f'Chain {chain_i + 1}', color='black', linewidth=5)
	# axis.set_xlim(0, n_generations)
	# axis.set_ylim(*pad_range(*MEASURE_RANGES[measure]))
	# axis.set_xticks(list(range(n_generations + 1)))
	# axis.set_xlabel('Generation')
	# axis.set_ylabel(LABELS[measure])
	# axis.set_title(LABELS[condition])


def plot_predictive(axis, trace, var, color):
	generation = np.arange(1, 10)
	az.plot_hdi(generation, trace.posterior[var], hdi_prob=0.95, smooth=False, color=color, fill_kwargs={'alpha': 0.25, 'linewidth': 0}, ax=axis)
	axis.plot(generation, trace.posterior[var].mean(('chain', 'draw')), color=color)
	# axis.set_ylim(0, 1.8)


df = pd.read_csv('../data/exp3.csv')
trace = az.from_netcdf('../data/exp1.netcdf')

fig, axes = plt.subplots(2, 2, figsize=(6, 4))

plot_chains(axes[0,0], df[ df['condition'] == 'dif_lrn' ], 'cost', show_mean=True)
plot_chains(axes[0,1], df[ df['condition'] == 'dif_com' ], 'cost', show_mean=True)

axes[0,0].set_ylim(0, 1.8)
axes[0,1].set_ylim(0, 1.8)

plot_predictive(axes[1,0], trace, 'pred_lrn', 'cadetblue')
plot_predictive(axes[1,1], trace, 'pred_com', 'crimson')

axes[1,0].set_ylim(0, 1.8)
axes[1,1].set_ylim(0, 1.8)

axes[1,0].set_xlim(0, 9)
axes[1,1].set_xlim(0, 9)

axes[1,0].set_xlabel('Generation')
axes[1,0].set_ylabel('Communicative cost (bits)')
axes[1,1].set_xlabel('Generation')
axes[1,1].set_ylabel('Communicative cost (bits)')

fig.tight_layout()

plt.show()