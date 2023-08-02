from pathlib import Path
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colormaps
import Levenshtein
import mantel
import grammarette

from utils import json_load, json_save


ROOT = Path(__file__).parent.parent.resolve()


COLORS = colormaps['tab10_r'].colors

MEASURE_RANGES = {
	'cost': (0, 2),
	'complexity': (100, 600),
	'structure': (0, 10),
	'error': (0, 3),
	'alignment': (0, 3),
}

LABELS = {
	'cost': 'Communicative cost (bits)',
	'complexity': 'Complexity (bits)',
	'structure': 'Structure (z-score)',
	'error': 'Transmission error (edits)',
	'alignment': 'Alignment (edits)',
	'lrn_hiVar': 'Learning-only',
	'lrn_noVar': 'Learning-only, No variation',
	'com_hiVar': 'Learning & Communication',
	'com_noVar': 'Learning & Communication, No variation',
	'lrn': 'Learning-only',
	'com': 'Learning & Communication',
}


def convert_lexicon_meanings_to_tuple(lexicon):
	converted_lexicon = {}
	for item, signal in lexicon.items():
		meaning = tuple(map(int, item.split('_')))
		converted_lexicon[meaning] = signal
	return converted_lexicon

def transmission_error(lexicon1, lexicon2):
	sum_lev_dist = 0.0
	for item, word1 in lexicon1.items():
		word2 = lexicon2[item]
		sum_lev_dist += Levenshtein.distance(word1, word2)
	return sum_lev_dist / len(lexicon1)

def complexity(lexicon, dims):
	grammar = grammarette.induce(lexicon, dims)
	return grammar.codelength

def structure(lexicon, dims):
	meaning_dists = []
	string_dists = []
	meanings, strings = zip(*lexicon.items())
	for i in range(len(meanings)):
		for j in range(i + 1, len(meanings)):
			meaning_dists.append(Levenshtein.distance(meanings[i], meanings[j]))
			string_dists.append(Levenshtein.distance(strings[i], strings[j]))
	meantel_result = mantel.test(meaning_dists, string_dists)
	return meantel_result.z

def communicative_cost(lexicon, dims):
	reverse_lexicon = defaultdict(set)
	for meaning, signal in lexicon.items():
		reverse_lexicon[signal].add(meaning)
	U = product(*[range(n_values) for n_values in dims])
	U_size = np.product(dims)
	return 1 / U_size * sum([-np.log2(1 / len(reverse_lexicon[lexicon[m]])) for m in U])

def perform_measures(exp_data_file, exp_csv_file):
	exp_data = json_load(exp_data_file)

	table = []
	for condition, data in exp_data.items():
		print(condition.upper())
		for chain_i, chain in enumerate(data):
			print('  Chain', chain_i)
			prev_lexicon = None
			for generation_i, (subject_a, subject_b) in enumerate(chain):
				print('    Generation', generation_i)
				lexicon_a = convert_lexicon_meanings_to_tuple(subject_a['lexicon'])
				error = transmission_error(lexicon_a, prev_lexicon) if prev_lexicon else None
				cost = communicative_cost(lexicon_a, (4, 4))
				# comp = complexity(lexicon_a, (4, 4))
				struc = structure(lexicon_a, (4, 4))
				if subject_b:
					lexicon_b = convert_lexicon_meanings_to_tuple(subject_b['lexicon'])
					algn = transmission_error(lexicon_a, lexicon_b)
				else:
					algn = None
				table.append([
					condition,
					chain_i,
					generation_i,
					error,
					struc,
					cost,
					algn,
				])
				prev_lexicon = lexicon_a

	df = pd.DataFrame(table, columns=['condition', 'chain', 'generation', 'error', 'structure', 'cost', 'alignment'])
	df.to_csv(exp_csv_file)


def pad_range(lo, hi):
	diff = hi - lo
	pad = diff * 0.05
	return lo - pad, hi + pad 

def plot_generational_change(axis, dataset, condition, measure, n_generations=20, show_mean=False):
	chain_data = []
	condition_subset = dataset[ dataset['condition'] == condition ]
	for chain_i in sorted(condition_subset['chain'].unique()):
		chain_subset = condition_subset[ condition_subset['chain'] == chain_i ]
		chain_data.append(list(chain_subset[measure]))
		if show_mean:
			continue
		axis.plot(chain_subset['generation'], chain_subset[measure], label=f'Chain {chain_i + 1}', color=COLORS[chain_i])
	if show_mean:
		chain_data = np.array(chain_data)
		axis.plot(chain_subset['generation'], chain_data.mean(axis=0), label=f'Chain {chain_i + 1}', color=COLORS[chain_i])
	axis.set_xlim(0, n_generations)
	axis.set_ylim(*pad_range(*MEASURE_RANGES[measure]))
	axis.set_xticks(list(range(n_generations + 1)))
	axis.set_xlabel('Generation')
	axis.set_ylabel(LABELS[measure])
	axis.set_title(LABELS[condition])
	# axis.legend()

def plot_generational_change_by_condition(dataset, measure):
	conditions = dataset['condition'].unique()
	conditions = list(reversed(sorted(conditions)))
	if len(conditions) == 2:
		fig, axes = plt.subplots(1, 2, figsize=(15, 7))
		n_generations = 20
	elif len(conditions) == 4:
		fig, axes = plt.subplots(2, 2, figsize=(15, 10))
		n_generations = 20
	for axis, condition in zip(np.ravel(axes), conditions):
		plot_generational_change(axis, dataset, condition, measure, n_generations, show_mean=False)
		# axis.plot(range(1, 13), [0, 0, 0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0], color='gray', linewidth=10, zorder=0)
		# axis.plot(range(1, 13), [209, 209, 209, 180, 180, 180, 206, 206, 206, 123, 123, 123], color='gray', linewidth=10, zorder=0)
		
	fig.tight_layout()
	plt.show()

def arrowplot(axis, X, Y, color=None):
	axis.scatter(X, Y, color=color)
	XY = np.column_stack([X, Y])
	for i in range(len(XY) - 1):
		start = XY[i]
		end = XY[i + 1]
		patch = patches.FancyArrowPatch(start, end, color=color, mutation_scale=10, alpha=0.3)
		axis.add_patch(patch)

def plot_simplicity_informativeness(axis, dataset, condition):
	condition_subset = dataset[ dataset['condition'] == condition ]
	for chain_i in sorted(condition_subset['chain'].unique()):
		chain_subset = condition_subset[ condition_subset['chain'] == chain_i ]
		arrowplot(axis, chain_subset['complexity'], chain_subset['cost'], COLORS[chain_i])
	axis.set_xlim(*pad_range(*MEASURE_RANGES['complexity']))
	axis.set_ylim(*pad_range(*MEASURE_RANGES['cost']))
	axis.set_xlabel(LABELS['complexity'])
	axis.set_ylabel(LABELS['cost'])
	axis.set_title(LABELS[condition])

def plot_simplicity_informativeness_by_condition(dataset):
	fig, axes = plt.subplots(2, 2, figsize=(15, 10))
	plot_simplicity_informativeness(axes[0,0], dataset, 'lrn_hiVar')
	plot_simplicity_informativeness(axes[0,1], dataset, 'com_hiVar')
	plot_simplicity_informativeness(axes[1,0], dataset, 'lrn_noVar')
	plot_simplicity_informativeness(axes[1,1], dataset, 'com_noVar')
	fig.tight_layout()
	plt.show()

def plot_training_curve(subject_id, window=12):
	if subject_id is None:
		return
	data = json_load(ROOT / 'data' / 'exp2' / f'subject_{subject_id}.json')
	fig, axis = plt.subplots(1, 1)
	full_correct = []
	for trial in data['responses']:
		if trial['test_type'] == 'mini_test':
			if trial['input_label'] == trial['expected_label']:
				full_correct.append(1)
			else:
				full_correct.append(0)

	x = []
	y = []
	for i in range(0, len(full_correct) - (window - 1)):
		mean = sum(full_correct[i : i + window]) / window
		x.append(i + window )
		y.append(mean)
	y = np.array(y)# + (np.random.random() * 0.1 - 0.05)
	axis.plot(x, y, color='black')
	axis.set_ylim(-0.05, 1.05)
	axis.set_xlim(1, 36)
	axis.set_xticks([1, 12, 24, 36])
	axis.set_xlabel('Mini-test trial')
	axis.set_ylabel(f'Mean accuracy over previous {window} trials')
	axis.axvline(12, color='black', linestyle='--')
	axis.axvline(24, color='black', linestyle='--')
	fig.savefig(ROOT / 'plots' / 'exp2' / f'learn_curve_{data["chain_id"]}_{data["generation"]}_{data["subject_id"]}.pdf')
	plt.close()
	# plt.show()

cons = ['θ', 's', 'ʃ', 'h']
vwls = ['ə', 'ɛɪ', 'əʊ', 'u']
def get_sound(item, data):
	sound_file = data['spoken_forms'][item]
	s, c, v = sound_file.split('.')[0].split('_')
	return f'{cons[int(c)]}{vwls[int(v)]}'.ljust(4)

def print_word_chains(dataset):
	for condition, data in dataset.items():
		print(condition.upper())
		for chain_i, chain in enumerate(data):
			print('  Chain', chain_i)
			table = [[] for _ in range(16)]
			for item_i, (shape, color) in enumerate(product(range(4), range(4))):
				item = f'{shape}_{color}'
				word = chain[0][0]['lexicon'][item]
				table[item_i].append(word.ljust(9, ' '))
				for subject_a, subject_b in chain[1:]:
					bottleneck = '➤ ' if item in subject_a['training_items'] else '  '
					word = subject_a['lexicon'][item]
					sound = get_sound(item, subject_a)
					table[item_i].append(bottleneck + sound + word.ljust(9, ' '))
			print(''.join([str(gen_i).ljust(16, ' ') for gen_i in range(len(table[0]))]).strip())
			for row in table:
				print(' '.join(row).strip())

def detect_convergence_generation(chain):
	prev_lexicon = chain[0][0]['lexicon']
	min_error = np.inf
	convergence_gen = None
	for gen_i, (subject_a, _) in enumerate(chain[1:], 1):
		error = transmission_error(prev_lexicon, subject_a['lexicon'])
		if error < min_error:
			min_error = error
			convergence_gen = gen_i
		prev_lexicon = subject_a['lexicon']
	return convergence_gen

import matrix
def draw_converged_matrixes(exp_data):
	for condition, data in exp_data.items():
		print(condition.upper())
		for chain_i, chain in enumerate(data):
			print('  Chain', chain_i)
			convergence_gen = detect_convergence_generation(chain)
			converged_lexicon = chain[convergence_gen][0]['lexicon']
			mat = matrix.make_matrix(converged_lexicon)
			cp = matrix.generate_color_palette(mat)
			matrix.draw(mat, cp, f'/Users/jon/Desktop/converged/{condition}_{chain_i}.pdf')

def draw_all_matrixes(exp_data):
	for condition, data in exp_data.items():
		print(condition.upper())
		for chain_i, chain in enumerate(data):
			print('  Chain', chain_i)
			cp = matrix.generate_color_palette_many(chain)
			for gen_i, (subject_a, _) in enumerate(chain):
				mat = matrix.make_matrix(subject_a['lexicon'])
				matrix.draw(mat, cp, f'/Users/jon/Desktop/matrices/{condition}_{chain_i}_{gen_i}.pdf')

import disttern
def make_ternary_plot():
	exp_data_file = ROOT / 'data' / 'exp1.json'
	exp_data = json_load(exp_data_file)
	ref_objects = [matrix.typology['transparent'], matrix.typology['redundant'], matrix.typology['expressive']]
	for condition, data in exp_data.items():
		scatter_objects = [matrix.make_matrix(chain[detect_convergence_generation(chain)][0]['lexicon']) for chain in data]
		for chain in data:
			for subject_a, _ in chain:
				scatter_objects.append(matrix.make_matrix(subject_a['lexicon']))
		disttern.make_ternary_plot(ref_objects, scatter_objects, matrix.voi.variation_of_information, jitter=True)

# def make_ternary_plot():
# 	exp_data_file = ROOT / 'data' / 'exp1.json'
# 	exp_data = json_load(exp_data_file)
# 	ref_objects = [matrix.typology['transparent'], matrix.typology['redundant'], matrix.typology['expressive']]
# 	lrn_matrices = [matrix.make_matrix(chain[detect_convergence_generation(chain)][0]['lexicon']) for chain in exp_data['lrn_hiVar']]
# 	com_matrices = [matrix.make_matrix(chain[detect_convergence_generation(chain)][0]['lexicon']) for chain in exp_data['com_hiVar']]
# 	disttern.make_ternary_plot(
# 		ref_objects,
# 		lrn_matrices + com_matrices,
# 		matrix.voi.variation_of_information,
# 		color=['CadetBlue'] * 10 + ['Crimson'] * 10,
# 		jitter=True,
# 	)

# def make_ternary_plot():
# 	exp_data_file = ROOT / 'data' / 'exp1.json'
# 	exp_data = json_load(exp_data_file)
# 	ref_objects = [matrix.typology['transparent'], matrix.typology['redundant'], matrix.typology['expressive']]
# 	lrn_matrices = []
# 	for n_suffixes in range(2, 16):
# 		lrn_matrices.extend( [np.random.randint(0, n_suffixes, 16).reshape((4,4)) for _ in range(1000)] )
# 	disttern.make_ternary_plot(
# 		ref_objects,
# 		lrn_matrices,
# 		matrix.voi.variation_of_information,
# 		color='CadetBlue',
# 		jitter=True,
# 	)

if __name__ == '__main__':

	exp_json_file = ROOT / 'data' / 'exp2.json'
	exp_csv_file = ROOT / 'data' / 'exp2.csv'

	# perform_measures(exp_json_file, exp_csv_file)

	dataset_json = json_load(exp_json_file)
	dataset_csv = pd.read_csv(exp_csv_file)

	plot_generational_change_by_condition(dataset_csv, 'error')
	# plot_simplicity_informativeness_by_condition(dataset_csv)

	# draw_converged_matrixes(dataset_json)
	# draw_all_matrixes(dataset_json)
	# make_ternary_plot()

	# print_word_chains(dataset_json)
