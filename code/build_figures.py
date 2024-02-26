from pathlib import Path
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as transforms
from matplotlib import colormaps
from scipy.stats import gaussian_kde
from utils import json_load


plt.rcParams.update({'font.sans-serif': 'Helvetica Neue', 'font.size': 7})


ROOT = Path(__file__).parent.parent.resolve()
DATA = ROOT / 'data'
FIGS = ROOT / 'manuscript' / 'figs'

COLORS = {
	'dif_lrn': 'darkolivegreen',
	'dif_com': 'darkorange',
	'con_lrn': 'cadetblue',
	'con_com': 'crimson',
	'sil_com': 'darkslateblue',
}
DISTRIBUTION_COLORS = colormaps['viridis'](np.linspace(0, 1, 4))

SINGLE_COLUMN = 3.54
DOUBLE_COLUMN = 7.48

HDI_PROB = 0.95


def plot_transmission_chains(exp_data, condition, output_path):
	import draw_chains
	import matrix

	cons = ['f', 's', 'ʃ']
	vwls = ['əʊ', 'ə', 'ɛɪ']
	def get_sound(item, data):
		sound_file = data['spoken_forms'][item]
		if '_' not in sound_file:
			return 'kəʊ'
		s, c, v = sound_file.split('.')[0].split('_')
		return f'{cons[int(c)]}{vwls[int(v)]}'

	A_to_J = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
	K_to_T = ['K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
	panel = []
	for chain in exp_data[condition]:
		chain_matrices = []
		ss, cp = matrix.generate_color_palette_many(chain)
		for subject_a, _ in chain:
			mat = matrix.make_matrix_with_cp(subject_a['lexicon'], ss, 3, 3)
			sounds = []
			if 'spoken_forms' in subject_a:
				if 'con' in condition:
					sounds = [get_sound(f'0_{i}', subject_a) for i in range(3)]
				elif 'dif' in condition:
					sounds = ['kəʊ', 'kəʊ', 'kəʊ']
			if 'training_items' in subject_a:
				training_items = subject_a['training_items']
			else:
				training_items = []
			chain_matrices.append((mat, cp, ss, sounds, training_items))
		panel.append(chain_matrices)
	show_sounds = True if 'con' in condition else False
	chain_ids = A_to_J if 'lrn' in condition or 'sil' in condition else K_to_T
	draw_chains.draw_panel(output_path, panel, chain_ids=chain_ids, show_sounds=show_sounds, figure_width=538)


def plot_typological_distribution(axis, distribution):
	categories = ['H', 'E', 'R', 'D']
	distribution = [distribution[category] for category in categories]
	axis.bar(categories, distribution, color=DISTRIBUTION_COLORS)
	axis.set_ylim(0, 10)
	axis.set_yticks([])
	

def plot_model(axis, trace, condition, condition_name, min_gen=0):
	axis.plot(trace.constant_data['x'], trace.posterior.y_model.mean(('chain', 'draw')), color=COLORS[condition], linewidth=3, label=condition_name)
	az.plot_hdi(
	    trace.constant_data['x'],
	    trace.posterior.y_model,
	    hdi_prob=0.95,
	    fill_kwargs={'alpha': 0.5, 'linewidth': 0},
	    ax=axis,
	    color=COLORS[condition],
	    smooth=False,
	)
	axis.set_xlabel('Generation')
	axis.set_ylabel('Proportion informative')
	# axis.set_ylim(0, 1)
	axis.set_xlim(min_gen - 0.25, 9 + 0.25)
	axis.set_xticks(range(min_gen, 10))


def plot_cost_model(axis, trace, condition, condition_name, min_gen=0):
	axis.plot(trace.constant_data['x'], trace.posterior.y_model.mean(('chain', 'draw')), color=COLORS[condition], linewidth=3, label=condition_name)
	az.plot_hdi(
	    trace.constant_data['x'],
	    trace.posterior.y_model,
	    hdi_prob=0.95,
	    fill_kwargs={'alpha': 0.5, 'linewidth': 0},
	    ax=axis,
	    color=COLORS[condition],
	    smooth=False,
	)
	axis.set_xlabel('Generation')
	# axis.set_ylim(0, 2)
	axis.set_xlim(min_gen - 0.25, 9 + 0.25)
	axis.set_xticks(range(min_gen, 10))


def plot_pinf_data(axis, data, condition, condition_name, min_gen=0):
	data = data[ (data['condition'] == condition) & (data['generation'] >= min_gen) ]
	prop_inf = data.groupby('generation')['informative'].mean()
	axis.scatter(np.arange(min_gen, 10), prop_inf, color=COLORS[condition], alpha=0.5, s=15, linewidth=0)


def plot_cost_data(axis, data, variable, condition, condition_name, min_gen=0):
	data = data[ (data['condition'] == condition) & (data['generation'] >= min_gen) ]
	for chain, cost_over_chain in data.groupby('chain')[variable]:
		axis.plot(range(min_gen, 10), cost_over_chain, color=COLORS[condition], alpha=0.25, zorder=-50)


def plot_posterior(axis, trace, param, condition):
	samples = trace.posterior[param].to_numpy().flatten()
	az_hdi = az.hdi(samples, hdi_prob=0.95)
	lower, upper = float(az_hdi[0]), float(az_hdi[1])
	x_min = lower - (upper - lower) / 2
	x_max = upper + (upper - lower) / 2
	x = np.linspace(x_min, x_max, 100)
	y = gaussian_kde(samples).pdf(x)
	axis.plot(x, y, color=COLORS[condition])
	axis.fill_between(x, np.zeros(len(y)), y, color=COLORS[condition], linewidth=0, alpha=0.5)
	axis.set_yticks([])
	# axis.set_xlabel(f'${param.replace("1", "_1").replace("2", "_2")}$')
	axis.text(0.03, 0.94, f'${param.replace("1", "_1").replace("2", "_2")}$', transform=axis.transAxes, ha='left', va='top')
	return x_min, x_max


def draw_hdis(axis, samples, hdi_probs):
	mn, mx = axis.get_ylim()
	stack_padding = (mx - mn) / 30
	colors = ['MediumSeaGreen', 'SteelBlue', 'Tomato']
	for i, hdi_prob in enumerate(hdi_probs):
		az_hdi = az.hdi(samples, hdi_prob=hdi_prob)
		lower = float(az_hdi[0])
		upper = float(az_hdi[1])
		mn_y, mx_y  = axis.get_ylim()
		padding = (mx_y - mn_y) * 0.1
		axis.plot((lower, upper), (i*stack_padding + stack_padding, i*stack_padding + stack_padding), color=colors[i], label=f'{int(round(hdi_prob*100, 0))}%')


def plot_posterior_difference(axis, trace1, trace2, param):
	samples_diff = (trace2.posterior[param] - trace1.posterior[param]).to_numpy().flatten()
	az_hdi = az.hdi(samples_diff, hdi_prob=0.95)
	lower, upper = float(az_hdi[0]), float(az_hdi[1])
	x_min = lower - (upper - lower) / 2
	x_max = upper + (upper - lower) / 2
	x = np.linspace(x_min, x_max, 100)
	y = gaussian_kde(samples_diff).pdf(x)
	axis.axvline(0, color='gray', linestyle=':')
	axis.plot(x, y, color='black')
	axis.fill_between(x, np.zeros(len(y)), y, color='gray', linewidth=0, alpha=0.25)
	draw_hdis(axis, samples_diff, [0.95, 0.90, 0.85])
	axis.set_yticks([])
	axis.set_xlim(x_min, x_max)
	axis.set_ylim(0, axis.get_ylim()[1])
	# axis.set_xlabel(f'$Δ({param.replace("1", "_1").replace("2", "_2")})$')
	axis.text(0.02, 0.94, f'$Δ({param.replace("1", "_1").replace("2", "_2")})$', transform=axis.transAxes, ha='left', va='top')


def abbreviate(label):
	if len(label) < 10:
		return label
	label = label.replace('Transmission', 'Trans.')
	label = label.replace('Communication', 'Comm.')
	return label


def plot_results(data, conditions, condition_names=None, params=[], show_epochs=False, output_path=None):
	if condition_names is None:
		condition_names = conditions

	fig = plt.figure(figsize=(DOUBLE_COLUMN, 7.5))

	gs = gridspec.GridSpec(8, 10, figure=fig)
	gs_typol_dist = gridspec.GridSpecFromSubplotSpec(2, 9, subplot_spec=gs[0:2, :])
	
	gs_model = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[2:4, 0:6])
	gs_posterior = gridspec.GridSpecFromSubplotSpec(2, len(params), subplot_spec=gs[2:4, 6:10])

	gs_cost_model = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[4:6, 0:6])
	gs_cost_posterior = gridspec.GridSpecFromSubplotSpec(2, len(params), subplot_spec=gs[4:6, 6:10])

	gs_error_model = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[6:8, 0:6])
	gs_error_posterior = gridspec.GridSpecFromSubplotSpec(2, len(params), subplot_spec=gs[6:8, 6:10])

	for i, condition in enumerate(conditions):
		data_condition = data[ data['condition'] == condition ]
		distributions = data_condition.groupby(['generation', 'type']).size().unstack(fill_value=0)
		for j, generation in enumerate(range(1, 10)):
			axis = fig.add_subplot(gs_typol_dist[i, j])
			plot_typological_distribution(axis, distributions.loc[generation])
			if i == 0:
				axis.set_title(f'Gen. {generation}', fontsize=7)
			if j == 0:
				axis.set_yticks(list(range(0, 11, 2)))
				axis.set_ylabel(abbreviate(condition_names[i]) + '\nCount')


	trace1 = az.from_netcdf(DATA / f'models/{conditions[0]}_pinf.netcdf')
	trace2 = az.from_netcdf(DATA / f'models/{conditions[1]}_pinf.netcdf')

	axis = fig.add_subplot(gs_model[0, 0])
	plot_pinf_data(axis, data, conditions[0], condition_names[0], min_gen=1)
	plot_pinf_data(axis, data, conditions[1], condition_names[1], min_gen=1)
	plot_model(axis, trace1, conditions[0], condition_names[0], min_gen=1)
	plot_model(axis, trace2, conditions[1], condition_names[1], min_gen=1)
	axis.legend(frameon=False)
	axis.set_xlabel(None)
	if show_epochs:
		axis.axvline(3.5, color='gray', linewidth=0.5, linestyle='--', zorder=-100)
		axis.axvline(6.5, color='gray', linewidth=0.5, linestyle='--', zorder=-100)
	
	for i, param in enumerate(params):

		axis = fig.add_subplot(gs_posterior[0, i])
		mn1, mx1 = plot_posterior(axis, trace1, param, conditions[0])
		mn2, mx2 = plot_posterior(axis, trace2, param, conditions[1])
		axis.set_xlim(min(mn1, mn2), max(mx1, mx2))
		axis.set_ylim(0, axis.get_ylim()[1])

		axis = fig.add_subplot(gs_posterior[1, i])
		plot_posterior_difference(axis, trace1, trace2, param)


	trace1 = az.from_netcdf(DATA / f'models/{conditions[0]}_cost.netcdf')
	trace2 = az.from_netcdf(DATA / f'models/{conditions[1]}_cost.netcdf')

	axis = fig.add_subplot(gs_cost_model[0, 0])

	plot_cost_data(axis, data, 'cost', conditions[0], condition_names[0], min_gen=1)
	plot_cost_data(axis, data, 'cost', conditions[1], condition_names[1], min_gen=1)

	plot_cost_model(axis, trace1, conditions[0], condition_names[0], min_gen=1)
	plot_cost_model(axis, trace2, conditions[1], condition_names[1], min_gen=1)
	axis.set_ylabel('Communicative cost (bits)')
	axis.set_xlabel(None)
	if show_epochs:
		axis.axvline(3.5, color='gray', linewidth=0.5, linestyle='--', zorder=-100)
		axis.axvline(6.5, color='gray', linewidth=0.5, linestyle='--', zorder=-100)
	
	for i, param in enumerate(params):

		axis = fig.add_subplot(gs_cost_posterior[0, i])
		mn1, mx1 = plot_posterior(axis, trace1, param, conditions[0])
		mn2, mx2 = plot_posterior(axis, trace2, param, conditions[1])
		axis.set_xlim(min(mn1, mn2), max(mx1, mx2))
		axis.set_ylim(0, axis.get_ylim()[1])

		axis = fig.add_subplot(gs_cost_posterior[1, i])
		plot_posterior_difference(axis, trace1, trace2, param)


	trace1 = az.from_netcdf(DATA / f'models/{conditions[0]}_error.netcdf')
	trace2 = az.from_netcdf(DATA / f'models/{conditions[1]}_error.netcdf')

	axis = fig.add_subplot(gs_error_model[0, 0])

	plot_cost_data(axis, data, 'error', conditions[0], condition_names[0], min_gen=1)
	plot_cost_data(axis, data, 'error', conditions[1], condition_names[1], min_gen=1)

	plot_cost_model(axis, trace1, conditions[0], condition_names[0], min_gen=1)
	plot_cost_model(axis, trace2, conditions[1], condition_names[1], min_gen=1)
	axis.set_ylabel('Transmission error (edits)')
	if show_epochs:
		axis.axvline(3.5, color='gray', linewidth=0.5, linestyle='--', zorder=-100)
		axis.axvline(6.5, color='gray', linewidth=0.5, linestyle='--', zorder=-100)
	
	for i, param in enumerate(params):

		axis = fig.add_subplot(gs_error_posterior[0, i])
		mn1, mx1 = plot_posterior(axis, trace1, param, conditions[0])
		mn2, mx2 = plot_posterior(axis, trace2, param, conditions[1])
		axis.set_xlim(min(mn1, mn2), max(mx1, mx2))
		axis.set_ylim(0, axis.get_ylim()[1])

		axis = fig.add_subplot(gs_error_posterior[1, i])
		plot_posterior_difference(axis, trace1, trace2, param)

	fig.subplots_adjust(left=0.065, right=0.99, top=0.97, bottom=0.05, hspace=0.4)
	fig.text(0.01, 0.975, 'A', size=10, font='Arial', weight='bold')
	fig.text(0.01, 0.72, 'B', size=10, font='Arial', weight='bold')
	fig.text(0.01, 0.48, 'C', size=10, font='Arial', weight='bold')
	fig.text(0.01, 0.24, 'D', size=10, font='Arial', weight='bold')

	fig.savefig(output_path)


def plot_success(data, conditions, titles, output_path):
	fig, axes = plt.subplots(1, 3, sharey=True, figsize=(DOUBLE_COLUMN, 2.5))
	for axis, condition, title in zip(axes, conditions, titles):
		data_condition = data[ (data['condition'] == condition) & (data['generation'] > 0) ]
		for chain, success_over_chain in data_condition.groupby('chain')['success']:
			axis.plot(range(1, 10), success_over_chain, color=COLORS[condition], alpha=0.5, zorder=10)
		axis.set_ylim(0, 1)
		axis.set_xticks(range(1, 10))
		axis.set_title(title, fontsize=7)
		if title == 'Experiment 1':
			axis.set_ylabel('Proportion of trials successful')
		axis.axhline(1/9, color='gray', linestyle=':')
		axis.axhline(1/3, color='gray', linestyle='--')
	fig.tight_layout()
	fig.savefig(output_path)


if __name__ == '__main__':

	# Create detailed transmission chain figures

	# data = json_load(ROOT / 'data' / 'exp.json')
	
	# plot_transmission_chains(data, 
	# 	condition='dif_lrn',
	# 	output_path=FIGS / 'dif_lrn.pdf',
	# )

	# plot_transmission_chains(data, 
	# 	condition='dif_com',
	# 	output_path=FIGS / 'dif_com.pdf',
	# )

	# plot_transmission_chains(data, 
	# 	condition='con_lrn',
	# 	output_path=FIGS / 'con_lrn.pdf',
	# )

	# plot_transmission_chains(data, 
	# 	condition='con_com',
	# 	output_path=FIGS / 'con_com.pdf',
	# )

	# plot_transmission_chains(data, 
	# 	condition='sil_com',
	# 	output_path=FIGS / 'sil_com.pdf',
	# )


	# Plot quantitative results

	data = pd.read_csv(DATA / 'exp.csv')

	# Experiment 1

	# plot_results(data,
	# 	conditions=('dif_lrn', 'dif_com'),
	# 	condition_names=('Transmission-only', 'Transmission + Communication'),
	# 	params=['α', 'β'],
	# 	output_path=FIGS / 'results_exp1.pdf',
	# )

	# # Experiment 2

	# plot_results(data,
	# 	conditions=('con_lrn', 'con_com'),
	# 	condition_names=('Transmission-only', 'Transmission + Communication'),
	# 	params=['α', 'β', 'γ'],
	# 	show_epochs=True,
	# 	output_path=FIGS / 'results_exp2.pdf',
	# )

	# # Experiment 3

	# plot_results(data,
	# 	conditions=('dif_com', 'sil_com'),
	# 	condition_names=('Experiment 1', 'Experiment 3'),
	# 	params=['α', 'β'],
	# 	output_path=FIGS / 'results_exp3.pdf',
	# )

	# # Communicative success

	plot_success(data,
		conditions=['dif_com', 'con_com', 'sil_com'],
		titles=['Experiment 1', 'Experiment 2', 'Experiment 3'],
		output_path=FIGS / 'success.pdf',
	)
