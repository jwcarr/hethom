from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from utils import json_load


plt.rcParams.update({'font.sans-serif': 'Helvetica Neue', 'font.size': 7})


ROOT = Path(__file__).parent.parent.resolve()
DATA = ROOT / 'data'
FIGS = ROOT / 'manuscript' / 'figs'


COLORS = {
	'dif_lrn': ('darkslategray', '#97A7A7'),
	'dif_com': ('darkorange', '#FFC680'),
	'con_lrn': ('darkslategray', '#97A7A7'),
	'con_com': ('darkorange', '#FFC680'),
}


def plot_transmission_chains(exp_data, condition, output_path):
	import visualize
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
	chain_ids = A_to_J if 'lrn' in condition else K_to_T
	visualize.draw_panel(output_path, panel, chain_ids=chain_ids, show_sounds=show_sounds, figure_width=432)


def plot_typological_distribution(exp_data, conditions, generations, output_path=None, figsize=(6, 2), probabilistic_classification=False):
	import voi
	import matrix

	ref_systems = [
		matrix.reference_systems['holistic'],
		matrix.reference_systems['expressive'],
		matrix.reference_systems['redundant'],
		matrix.reference_systems['transparent'],
	]

	distribution_colors = colormaps['viridis'](np.linspace(0, 1, 4))

	fig, axes = plt.subplots(len(conditions), len(generations), figsize=figsize, squeeze=False, sharex=True, sharey=True)
	
	for i, condition in enumerate(conditions):
		for j, generation in enumerate(generations):
			distribution = np.zeros(len(ref_systems))
			for chain in exp_data[condition]:
				subject_a = chain[generation][0]
				subject_system = matrix.make_matrix(subject_a['lexicon'], 3, 3)
				distances = np.array([
					voi.variation_of_information(subject_system, ref_system)
					for ref_system in ref_systems
				])
				if probabilistic_classification:
					# -2 is weighting (free parameter), this value works well
					# because it results in almost all prob mass in
					# generation 0 being dedicated to the holistic category,
					# which we know to be correct.
					unnorm_distribution = np.exp(-2 * distances**2)
					distribution += unnorm_distribution / unnorm_distribution.sum()
				else:
					classification = np.where(distances == distances.min())[0]
					distribution[classification] += 1 / len(classification)
			distribution /= distribution.sum()
			axes[i,j].bar(['H', 'E', 'R', 'D'], distribution, color=distribution_colors)
			if i == 0:
				axes[i,j].set_title(f'Gen. {generation}', fontsize=7)
			if j == 0:
				if 'lrn' in condition:
					axes[0,0].set_ylabel('Trans.-only', fontsize=7)
				if 'com' in condition:
					axes[1,0].set_ylabel('Trans. + Comm.', fontsize=7)
	
	axes[0,0].set_ylim(0, 1)
	axes[0,0].set_yticks([])
	fig.tight_layout(pad=1, h_pad=1, w_pad=1)
	if output_path:
		fig.savefig(output_path)
	else:
		plt.show()


def plot_communicative_cost(exp_data, conditions, output_path=None, figsize=(6, 2), show_mean=True, add_jitter=False, model_trace=False, spoken_cost=None):
	import comm_cost
	if model_trace:
		import arviz as az
		trace = az.from_netcdf(model_trace)

	fig, axes = plt.subplots(1, len(conditions), figsize=figsize, squeeze=False, sharex=True, sharey=True)

	for i, condition in enumerate(conditions):

		cost_by_chain = []
		for chain in exp_data[condition]:
			cost_by_generation = []
			for gen_i, (subject_a, _) in enumerate(chain):
				cost_by_generation.append(
					comm_cost.communicative_cost(subject_a['lexicon'], dims=(3, 3))
				)
			cost_by_chain.append(cost_by_generation)
		cost_by_chain = np.array(cost_by_chain)

		color, color_light = COLORS[condition]

		for chain in cost_by_chain:
			if add_jitter:
				jitter = cost_by_chain.max() * 0.025
				axes[0,i].plot(chain + (np.random.random(len(chain)) - 0.5) * jitter, color=color, linewidth=1)
			else:
				axes[0,i].plot(chain)

		if model_trace:
			pred_var = f'pred_{condition.split("_")[1]}'
			
			pred = [trace.posterior[pred_var+f'_{i}'].mean(('chain', 'draw')) for i in range(1, 10)]
			pred_upper = [float(az.hdi(trace.posterior[pred_var+f'_{i}'].to_numpy().flatten(), hdi_prob=0.95)[1]) for i in range(1, 10)]
			pred_lower = [float(az.hdi(trace.posterior[pred_var+f'_{i}'].to_numpy().flatten(), hdi_prob=0.95)[0]) for i in range(1, 10)]
			
			axes[0,i].fill_between(range(1, 10), pred_lower, pred_upper, color=color_light, zorder=1.5)
			axes[0,i].plot(range(1, 10), pred, color=color, linewidth=4)

		if spoken_cost:
			axes[0,i].plot(range(0, 10), spoken_cost, color='black', linestyle=':', linewidth=2)

		if show_mean:
			axes[0,i].plot(cost_by_chain.mean(axis=0), color=color, linewidth=4)
		if 'lrn' in condition:
			axes[0,i].set_title('Transmission-only', fontsize=7)
		if 'com' in condition:
			axes[0,i].set_title('Transmission + Communication', fontsize=7)

	axes[0,0].set_xlabel('Generation')
	axes[0,1].set_xlabel('Generation')
	axes[0,0].set_ylabel('Communicative cost (bits)')
	axes[0,0].set_xticks(range(10))
	axes[0,1].set_xticks(range(10))
	axes[0,0].set_ylim(-0.15, 1.75)

	fig.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
	if output_path:
		fig.savefig(output_path)
	else:
		plt.show()


def draw_hdi(axis, lower, upper, hdi_prob):
	mn_y, mx_y  = axis.get_ylim()
	padding = (mx_y - mn_y) * 0.1
	axis.plot((lower, upper), (0, 0), color='MediumSeaGreen')
	hdi_text = f'{int(hdi_prob*100)}% HDI'
	hdi_width = round(upper - lower, 2)
	axis.text((lower + upper)/2, mn_y + padding, hdi_text, ha='center', color='MediumSeaGreen', fontsize=6)

def plot_posterior(model_trace, variables, output_path=None, figsize=(6, 2), show_summary=False):
	import arviz as az
	from scipy.stats import gaussian_kde

	trace = az.from_netcdf(model_trace)

	if show_summary:
		import pandas as pd
		pd.set_option('display.width', 200)
		pd.set_option('display.max_columns', None)
		pd.set_option('display.max_rows', None)
		table = az.summary(trace, hdi_prob=0.95)
		print(table)

	fig, axes = plt.subplots(2, len(variables), figsize=figsize, squeeze=False)

	for i, variable in enumerate(variables):

		if 'point_of_interest' in variable:
			axes[0,i].axvline(variable['point_of_interest'], color='gray', linestyle=':', linewidth=1)

		if 'sub' in variable:
			samples = trace.posterior[variable['var']].sel(epoch=variable['sub']).to_numpy().flatten()
		else:
			samples = trace.posterior[variable['var']].to_numpy().flatten()
		az_hdi = az.hdi(samples, hdi_prob=0.95)
		lower, upper = float(az_hdi[0]), float(az_hdi[1])
		x_min = lower - (upper - lower) / 2
		x_max = upper + (upper - lower) / 2

		x = np.linspace(x_min, x_max, 200)

		if 'sub' in variable:
			samples_lrn = trace.posterior[variable['var']].sel(condition='lrn', epoch=variable['sub']).to_numpy().flatten()
			y_lrn = gaussian_kde(samples_lrn).pdf(x)
			axes[0,i].plot(x, y_lrn, color=COLORS['dif_lrn'][0])

			samples_com = trace.posterior[variable['var']].sel(condition='com', epoch=variable['sub']).to_numpy().flatten()
			y_com = gaussian_kde(samples_com).pdf(x)
			axes[0,i].plot(x, y_com, color=COLORS['dif_com'][0])
		else:
			samples_lrn = trace.posterior[variable['var']].sel(condition='lrn').to_numpy().flatten()
			y_lrn = gaussian_kde(samples_lrn).pdf(x)
			axes[0,i].plot(x, y_lrn, color=COLORS['dif_lrn'][0])

			samples_com = trace.posterior[variable['var']].sel(condition='com').to_numpy().flatten()
			y_com = gaussian_kde(samples_com).pdf(x)
			axes[0,i].plot(x, y_com, color=COLORS['dif_com'][0])

		axes[0,i].set_yticks([])
		axes[0,i].set_xlim(x_min, x_max)
		axes[0,i].set_xlabel(variable['label'])

		if variable['diff']:

			samples_diff = trace.posterior[variable['diff']].to_numpy().flatten()
			az_hdi = az.hdi(samples_diff, hdi_prob=0.95)
			mx = max(abs(float(az_hdi[0])), abs(float(az_hdi[1]))) * 1.5
			lower, upper = float(az_hdi[0]), float(az_hdi[1])
			x_diff_min = lower - (upper - lower) / 2
			x_diff_max = upper + (upper - lower) / 2

			x_diff = np.linspace(x_diff_min, x_diff_max, 200)
			y_diff = gaussian_kde(samples_diff).pdf(x_diff)

			axes[1,i].plot(x_diff, y_diff, color='black')
			axes[1,i].set_yticks([])
			axes[1,i].set_xlim(x_diff_min, x_diff_max)
			axes[1,i].set_xlabel(f'Δ({variable["label"]})') 

			az_hdi = az.hdi(samples_diff, hdi_prob=0.95)
			draw_hdi(axes[1,i], float(az_hdi[0]), float(az_hdi[1]), 0.95)

		else:
			axes[1,i].axis('off')

	fig.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
	if output_path:
		fig.savefig(output_path)
	else:
		plt.show()


def mass_below_zero(model_trace, variables):
	import arviz as az

	trace = az.from_netcdf(model_trace)
	for variable in variables:
		posterior_samples = trace.posterior[variable].to_numpy().flatten()
		proportion = (posterior_samples < 0).sum() / len(posterior_samples)
		print(variable, proportion)


if __name__ == '__main__':

	exp_data = json_load(ROOT / 'data' / 'exp3.json')

	# plot_transmission_chains(exp_data, 
	# 	condition='dif_lrn',
	# 	output_path=FIGS / 'dif_lrn.eps',
	# )

	# plot_transmission_chains(exp_data, 
	# 	condition='dif_com',
	# 	output_path=FIGS / 'dif_com.eps',
	# )

	# plot_transmission_chains(exp_data, 
	# 	condition='con_lrn',
	# 	output_path=FIGS / 'con_lrn.eps',
	# )

	# plot_transmission_chains(exp_data, 
	# 	condition='con_com',
	# 	output_path=FIGS / 'con_com.eps',
	# )

	# plot_typological_distribution(exp_data,
	# 	conditions=['dif_lrn', 'dif_com'],
	# 	generations=list(range(10)),
	# 	output_path=FIGS / 'typ_dist_dif.eps',
	# 	figsize=(6, 2),
	# 	probabilistic_classification=True,
	# )

	# plot_typological_distribution(exp_data,
	# 	conditions=['con_lrn', 'con_com'], 
	# 	generations=list(range(10)),
	# 	output_path=FIGS / 'typ_dist_con.eps',
	# 	figsize=(6, 2),
	# 	probabilistic_classification=True,
	# )

	# plot_communicative_cost(exp_data,
	# 	conditions=['dif_lrn', 'dif_com'],
	# 	output_path=FIGS / 'cost_dif.eps',
	# 	figsize=(6, 2),
	# 	show_mean=False,
	# 	add_jitter=True,
	# 	model_trace=DATA / 'exp1.netcdf',
	# 	spoken_cost=[1.58496] * 10,
	# )

	# plot_posterior(DATA / 'exp1.netcdf',
	# 	variables=[
	# 		{'var': 'α_m',  'label': '$α$',   'diff': 'diff_α', 'point_of_interest': 1.58496},
	# 		{'var': 'β1_m', 'label': '$β_1$', 'diff': 'diff_β1'},
	# 		{'var': 'β2_m', 'label': '$β_2$', 'diff': 'diff_β2'},
	# 		{'var': 'β3_m', 'label': '$β_3$', 'diff': 'diff_β3'},
	# 	],
	# 	output_path=FIGS / 'posterior_dif.eps',
	# 	figsize=(6, 2.5),
	# 	show_summary=True,
	# )

	# plot_communicative_cost(exp_data,
	# 	conditions=['con_lrn', 'con_com'],
	# 	output_path=FIGS / 'cost_con.eps',
	# 	figsize=(6, 2),
	# 	show_mean=False,
	# 	add_jitter=True,
	# 	model_trace=DATA / 'exp2.netcdf',
	# 	spoken_cost=[0, 0, 0, 0, 0.66666, 0.66666, 0.66666, 1.58496, 1.58496, 1.58496],
	# )

	# plot_posterior(DATA / 'exp2.netcdf',
	# 	variables=[
	# 		{'var': 'α_m', 'sub':1, 'label': '$α_1$', 'diff': 'diff_α1', 'point_of_interest': 0.0},
	# 		{'var': 'α_m', 'sub':2, 'label': '$α_2$', 'diff': 'diff_α2', 'point_of_interest': 0.66666},
	# 		{'var': 'α_m', 'sub':3, 'label': '$α_3$', 'diff': 'diff_α3', 'point_of_interest': 1.58496},
	# 		{'var': 'β_m', 'sub':1, 'label': '$β_1$', 'diff': 'diff_β1'},
	# 		{'var': 'β_m', 'sub':2, 'label': '$β_2$', 'diff': 'diff_β2'},
	# 		{'var': 'β_m', 'sub':3, 'label': '$β_3$', 'diff': 'diff_β3'},
	# 	],
	# 	output_path=FIGS / 'posterior_con.eps',
	# 	figsize=(6, 2.5),
	# 	show_summary=True,
	# )

	# mass_below_zero(
	# 	model_trace=DATA / 'exp2.netcdf',
	# 	variables=['diff_α3', 'diff_β2']
	# )
