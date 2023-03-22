from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import mantel
import matrix


ROOT = Path(__file__).parent.parent.resolve()


def json_load(filepath):
	with open(filepath) as file:
		data = json.load(file)
	return data

def levenshtein_distance(s1, s2, normalize=False):
	if len(s1) > len(s2):
		s1, s2 = s2, s1
	distances = range(len(s1) + 1)
	for index2, char2 in enumerate(s2):
		newDistances = [index2 + 1]
		for index1, char1 in enumerate(s1):
			if char1 == char2:
				newDistances.append(distances[index1])
			else:
				newDistances.append(
					1 + min((distances[index1], distances[index1+1], newDistances[-1]))
				)
		distances = newDistances
	if normalize:
		return distances[-1] / max(len(s1), len(s2))
	return distances[-1]

def transmission_error(lexicon1, lexicon2):
	sum_norm_lev_dist = 0.0
	for item, word1 in lexicon1.items():
		word2 = lexicon2[item]
		sum_norm_lev_dist += levenshtein_distance(word1, word2, normalize=False)
	return sum_norm_lev_dist / len(lexicon1)

def expressivity(lexicon):
	return len(set(lexicon.values()))

def structure(lexicon):
	meaning_dists = []
	string_dists = []
	meanings, strings = zip(*lexicon.items())
	for i in range(len(meanings)):
		for j in range(i + 1, len(meanings)):
			meaning_dists.append(levenshtein_distance(meanings[i], meanings[j]))
			string_dists.append(levenshtein_distance(strings[i], strings[j]))
	meantel_result = mantel.test(meaning_dists, string_dists)
	return meantel_result.z

def plot_response_time(dataset):
	res = defaultdict(list)
	for chain, generations in dataset.items():
		for subject in generations:
			for trial in subject['responses']:
				res[trial['test_type']].append(trial['reaction_time'] / 1000)
	res_mt = res['mini_test']
	res_tp = res['test_production']
	res_tc = res['test_comprehension']
	plt.hist(res_mt, bins=np.linspace(0, 60, 61))
	plt.xlim(0, 60)
	plt.xlabel('Response time (seconds)')
	plt.show()

def plot_transmission_error(dataset):
	for chain, generations in dataset.items():
		error = []
		for generation in generations:
			error.append(
				transmission_error(generation['input_lexicon'], generation['lexicon'])
			)
		plt.plot(range(1, len(error) + 1), error)
		plt.xlim(0, 5)
		plt.ylim(0, 3)
		plt.xlabel('Generation')
		plt.ylabel('Transmission error')
	plt.show()

def plot_expressivity(dataset):
	for chain, generations in dataset.items():
		expr = []
		gen0 = {'lexicon': generations[0]['input_lexicon']}
		for generation in [gen0] + generations:
			expr.append(
				expressivity(generation['lexicon'])
			)
		plt.plot(expr)
		plt.xlim(0, 5)
		plt.ylim(0, 16)
		plt.xlabel('Generation')
		plt.ylabel('Expressivity')
	plt.show()

def plot_structure(dataset):
	for chain, generations in dataset.items():
		expr = []
		gen0 = {'lexicon': generations[0]['input_lexicon']}
		for generation in [gen0] + generations:
			expr.append(
				structure(generation['lexicon'])
			)
		plt.plot(expr)
		plt.xlim(0, 5)
		plt.ylim(4, 6.2)
		plt.xlabel('Generation')
		plt.ylabel('Structure')
	plt.show()

def scramble(word):
	word_list = list(word)
	np.random.shuffle(word_list)
	return ''.join(word_list)

def baseline_structure():
	sys_lang = {
		"0_0": "buvikop",
		"0_1": "buvikog",
		"0_2": "buvikob",
		"0_3": "buvikoy",
		"1_0": "zetikop",
		"1_1": "zetikog",
		"1_2": "zetikob",
		"1_3": "zetikoy",
		"2_0": "gafikop",
		"2_1": "gafikog",
		"2_2": "gafikob",
		"2_3": "gafikoy",
		"3_0": "wopikop",
		"3_1": "wopikog",
		"3_2": "wopikob",
		"3_3": "wopikoy",
	}
	print(structure(sys_lang))
	unsys_lang = {item: scramble(word) for item, word in sys_lang.items()}
	scrambled_strings = list(unsys_lang.values())
	np.random.shuffle(scrambled_strings)
	unsys_lang = dict(zip(unsys_lang.keys(), scrambled_strings))
	print(unsys_lang)
	print(structure(unsys_lang))
	semi = {
		"0_0": "buvicow",
		"0_1": "buvychoh",
		"0_2": "buvikow",
		"0_3": "buvychow",
		"1_0": "zeteechoe",
		"1_1": "zetikoh",
		"1_2": "zeteekoh",
		"1_3": "zeteecoh",
		"2_0": "gafykoh",
		"2_1": "gafychoe",
		"2_2": "gafeecow",
		"2_3": "gafichoe",
		"3_0": "wopeekow",
		"3_1": "wopeechoh",
		"3_2": "wopichow",
		"3_3": "wopeekoe",
	}
	print(structure(semi))
	sys_uninf_lang = {
		"0_0": "buviko",
		"0_1": "buviko",
		"0_2": "buviko",
		"0_3": "buviko",
		"1_0": "zetiko",
		"1_1": "zetiko",
		"1_2": "zetiko",
		"1_3": "zetiko",
		"2_0": "gafiko",
		"2_1": "gafiko",
		"2_2": "gafiko",
		"2_3": "gafiko",
		"3_0": "wopiko",
		"3_1": "wopiko",
		"3_2": "wopiko",
		"3_3": "wopiko",
	}
	print(structure(sys_uninf_lang))


def print_word_chains(dataset):
	for chain, generations in dataset.items():
		for item, word in generations[0]['input_lexicon'].items():
			print(item, word)
		for generation in generations:
			print('-------------')
			for item, word in generation['lexicon'].items():
				bottleneck = '➡️' if item in generation['training_items'] else '  '
				print(bottleneck, word)

def print_comments(dataset):
	for chain, generations in dataset.items():
		for generation in generations:
			print(generation['subject_id'], generation['comments'])

import mds
def draw_matrices(dataset):
	for chain, generations in dataset.items():
		output_path = ROOT / 'plots' / chain
		if not output_path.exists():
			output_path.mkdir(parents=True)
		gen0 = {'lexicon': generations[0]['input_lexicon']}
		generations = [gen0] + generations
		suffix_spellings = []
		for gen in generations:
			for word in gen['lexicon'].values():
				suffix_spellings.append(word[3:])
		suffix_spellings = sorted(list(set(suffix_spellings)))
		cp = generate_color_palette(suffix_spellings)
		for gen, generation in enumerate(generations):
			mat, _ = matrix.make_matrix(generation['lexicon'], suffix_spellings)
			print(mat)
			print(model.posterior_over_grammars(mat))
			matrix.draw(mat, cp, str(output_path / f'{gen}.pdf'))
		
def generate_color_palette(suffix_spellings):
	n_spellings = len(suffix_spellings)
	hues = list(np.linspace(0, 2 * np.pi, n_spellings))
	return {suffix_spellings.index(suffix): hsv_to_rgb(hues.pop(), 0.8, 0.8) for suffix in suffix_spellings}

# Convert hue [0,2pi], saturation [0,1], and brightness [0,1] into RGB
def hsv_to_rgb(h, s, v):
	if s == 0.0: return v, v, v # saturation is 0, so return white
	h /= 2 * np.pi # scale hue (expressed in radians) in [0,1]
	i = int(h*6.)
	f = (h*6.)-i
	p, q, t = v*(1.-s), v*(1.-s*f), v*(1.-s*(1.-f))
	i %= 6
	if i == 0: return v, t, p
	elif i == 1: return q, v, p
	elif i == 2: return p, v, t
	elif i == 3: return p, q, v
	elif i == 4: return t, p, v
	return v, p, q

def plot_typ_dist(distribution):
	fig, axis = plt.subplots(1, 1)
	axis.bar(['T', 'H', 'R', 'E'], distribution)
	axis.set_ylim(0, 1)
	return fig, axis

def typ_dist(dataset):
	for chain, generations in dataset.items():
		for gen_i, generation in enumerate(generations):
			mat = matrix.make_matrix(generation['lexicon'])
			# print(mat)
			dist = matrix.typological_distribution(mat, probabilistic=True)
			fig, axis = plot_typ_dist(dist)
			axis.set_title(chain + str(gen_i))
			plt.show()


def training_curve(dataset, window=12):
	fig, axis = plt.subplots(1, 1)
	for chain, generations in dataset.items():
		for gen_i, generation in enumerate(generations):
			full_correct = []
			part_correct = []
			for trial in generation['responses']:
				if trial['test_type'] == 'mini_test':
					if trial['input_label'] == trial['expected_label']:
						full_correct.append(1)
					else:
						full_correct.append(0)
					if trial['input_label'][:3] == trial['expected_label'][:3]:
						part_correct.append(1)
					else:
						part_correct.append(0)			
			x = []
			y = []
			for i in range(0, len(full_correct) - (window - 1)):
				mean = sum(full_correct[i : i + window]) / window
				x.append(i + window )
				y.append(mean)
			y = np.array(y) + (np.random.random() * 0.1 - 0.05)
			axis.plot(x, y)
	axis.set_ylim(-0.05, 1.05)
	axis.set_xlim(1, 36)
	axis.set_xticks([1, 12, 24, 36])
	axis.set_xlabel('Mini-test trial')
	axis.set_ylabel(f'Mean accuracy over previous {window} trials')
	axis.axvline(12, color='black', linestyle='--')
	axis.axvline(24, color='black', linestyle='--')
	plt.show()

					
					# correct.append(trial['correct'])


dataset = json_load(ROOT / 'data' / 'pilot1.json')

# plot_response_time(dataset)
# plot_transmission_error(dataset)
# plot_expressivity(dataset)
# plot_structure(dataset)
# baseline_structure()
# print_word_chains(dataset)
# print_comments(dataset)
# draw_matrices(dataset)
# typ_dist(dataset)
training_curve(dataset)



