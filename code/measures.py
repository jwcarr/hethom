from collections import defaultdict
from itertools import product
from pathlib import Path
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mantel
import matrix
import grammarette


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
		for subject_a, subject_b in generations:
			for trial in subject_a['responses']:
				res[trial['test_type']].append(trial['response_time'] / 1000)
			if subject_b:
				for trial in subject_b['responses']:
					res[trial['test_type']].append(trial['response_time'] / 1000)
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
		for subject_a, subject_b in generations:
			error.append(
				transmission_error(subject_a['input_lexicon'], subject_a['lexicon'])
			)
		plt.plot(range(1, len(error) + 1), error, label=chain)
		plt.xlim(0, 10)
		plt.ylim(-0.05, 3.05)
		plt.xlabel('Generation')
		plt.ylabel('Transmission error')
		plt.legend()
	plt.show()

def plot_expressivity(dataset):
	for chain, generations in dataset.items():
		expr = []
		gen0 = [{'lexicon': generations[0][0]['input_lexicon']}] * 2
		for subject_a, subject_b in [gen0] + generations:
			expr.append(
				expressivity(subject_a['lexicon'])
			)
		plt.plot(expr, label=chain)
		plt.xlim(0, 10)
		plt.ylim(0, 16)
		plt.xlabel('Generation')
		plt.ylabel('Expressivity')
		plt.legend()
	plt.show()

def convert_lexicon_meanings_to_tuple(lexicon):
	converted_lexicon = {}
	for item, signal in lexicon.items():
		meaning = tuple(map(int, item.split('_')))
		converted_lexicon[meaning] = signal
	return converted_lexicon

def cost(lexicon, dims):
	lexicon = convert_lexicon_meanings_to_tuple(lexicon)
	reverse_lexicon = defaultdict(set)
	for meaning, signal in lexicon.items():
		reverse_lexicon[signal].add(meaning)
	U = product(*[range(n_values) for n_values in dims])
	U_size = np.product(dims)
	return 1 / U_size * sum([-np.log2(1 / len(reverse_lexicon[lexicon[m]])) for m in U])

def plot_informativeness(dataset):
	for chain, generations in dataset.items():
		expr = []
		gen0 = [{'lexicon': generations[0][0]['input_lexicon']}] * 2
		for subject_a, subject_b in [gen0] + generations:
			expr.append(
				cost(subject_a['lexicon'], (4, 4))
			)
		plt.plot(expr, label=chain)
		plt.xlim(0, 10)
		plt.ylim(0, 2)
		plt.xlabel('Generation')
		plt.ylabel('Communicative cost (bits)')
		plt.legend()
	plt.show()

def complexity(lexicon, dims):
	lexicon = convert_lexicon_meanings_to_tuple(lexicon)
	grammar = grammarette.induce(lexicon, dims)
	return grammar.codelength

def plot_simplicity(dataset):
	for chain, generations in dataset.items():
		expr = []
		gen0 = [{'lexicon': generations[0][0]['input_lexicon']}] * 2
		for subject_a, subject_b in [gen0] + generations:
			expr.append(
				complexity(subject_a['lexicon'], (4, 4))
			)
		plt.plot(expr, label=chain)
		plt.xlim(0, 10)
		plt.xlabel('Generation')
		plt.ylabel('Complexity (bits)')
		plt.legend()
	plt.show()

def arrowplot(axis, X, Y, color=None):
	plt.scatter(X, Y, color=color)
	XY = np.column_stack([X, Y])
	for i in range(len(XY) - 1):
		start = XY[i]
		end = XY[i + 1]
		patch = patches.FancyArrowPatch(start, end, color=color, mutation_scale=10, alpha=0.3)
		axis.add_patch(patch)

def plot_simplicity_informativeness(dataset):
	simp = [
		[573.1087206239592, 538.7026969437203, 413.0861637257097, 470.60543986551664, 431.70531792231776, 393.11068992214194, 335.0913617625357, 439.3275789848815, 309.92194229384563, 271.965276246367, 323.35493941436675],
		[591.6751444173998, 440.34198518156006, 383.2835671103883, 466.5813837111495, 340.6108966905434, 359.42147265972216, 396.101151026367, 482.4223830616061, 338.9923319076298, 219.26353873315173, 139.19929909169417],
		[592.1867711964184, 457.8865826418728, 508.9304804657626, 430.04702999900263, 384.0064793733731, 466.6081489699085, 493.3217795544788, 500.9404769518681, 351.99432958416725, 230.2261782820019, 230.2261782820019],
	]
	infm = [
		[0.0, 0.125, 0.625, 0.375, 0.375, 0.5, 0.25, 0.7971804688852169, 1.1721804688852169, 1.5943609377704338, 1.3915414066556508],
		[0.0, 0.5, 1.0943609377704338, 0.7193609377704335, 0.7971804688852168, 0.6721804688852169, 1.0471804688852169, 1.1721804688852169, 1.0, 1.75, 2.0],
		[0.0, 0.25, 0.375, 0.7193609377704336, 0.625, 0.9693609377704335, 0.7971804688852168, 0.4221804688852168, 0.625, 0.5, 0.5],
	]
	fig, axis = plt.subplots(1, 1)
	for s, i in zip(simp, infm):
		arrowplot(axis, s, i, color='MediumSeaGreen')
	axis.set_xlim(75, 625)
	axis.set_ylim(-0.1, 2.1)
	axis.set_xlabel('Complexity (bits)')
	axis.set_ylabel('Cost (bits)')
	plt.show()


def plot_structure(dataset):
	for chain, generations in dataset.items():
		expr = []
		gen0 = [{'lexicon': generations[0][0]['input_lexicon']}] * 2
		for subject_a, subject_b in [gen0] + generations:
			expr.append(
				structure(subject_a['lexicon'])
			)
		plt.plot(expr, label=chain)
		plt.xlim(0, 10)
		plt.xlabel('Generation')
		plt.ylabel('Structure')
		plt.legend()
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
		print('-------------')
		print(chain)
		print('-------------')
		table = [[] for _ in range(16)]
		for item_i, (shape, color) in enumerate(product(range(4), range(4))):
			item = f'{shape}_{color}'
			word = generations[0][0]['input_lexicon'][item]
			table[item_i].append(word.ljust(9, ' '))
			for subject_a, subject_b in generations:
				bottleneck = '➤ ' if item in subject_a['training_items'] else '  '
				word = subject_a['lexicon'][item]
				table[item_i].append(bottleneck + word.ljust(9, ' '))
		print(''.join([str(gen_i).ljust(12, ' ') for gen_i in range(len(table[0]))]).strip())
		for row in table:
			print(' '.join(row).strip())

def print_comments(dataset):
	for chain, generations in dataset.items():
		for gen_i, (subject_a, subject_b) in enumerate(generations, 1):
			if subject_b:
				print(chain, gen_i, 'A', subject_a['comments'])
				print(chain, gen_i, 'B', subject_b['comments'])
			else:
				print(chain, gen_i, subject_a['comments'])

def draw_matrices(dataset):
	for chain, generations in dataset.items():
		output_path = ROOT / 'plots' / chain
		if not output_path.exists():
			output_path.mkdir(parents=True)
		gen0 = [{'lexicon': generations[0][0]['input_lexicon']}] * 2
		generations = [gen0] + generations
		suffix_spellings = []
		for gen in generations:
			for word in gen[0]['lexicon'].values():
				suffix_spellings.append(word[3:])
		suffix_spellings = sorted(list(set(suffix_spellings)))
		cp = generate_color_palette(suffix_spellings)
		for gen, generation in enumerate(generations):
			mat = matrix.make_matrix(generation[0]['lexicon'], suffix_spellings)
			matrix.draw(mat, cp, str(output_path / f'{gen}.pdf'))
		
def generate_color_palette(suffix_spellings):
	n_spellings = len(suffix_spellings)
	hues = list(np.linspace(0, 2 * np.pi, n_spellings + 1))
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
		for gen_i, (subject_a, subject_b) in enumerate(generations):
			mat = matrix.make_matrix(subject_a['lexicon'])
			dist = matrix.typological_distribution(mat, probabilistic=True)
			fig, axis = plot_typ_dist(dist)
			axis.set_title(chain + str(gen_i))
			plt.show()

chain_name_colors = {
	'com_hiVar_0':'blue',
	'com_noVar_0':'purple',
	'lrn_hiVar_0':'black',
	'lrn_noVar_0':'orange',
	'lrn_redun_0':'green',
	'lrn_exprs_0':'red',
}
def training_curve(dataset, window=12):
	fig, axis = plt.subplots(1, 1)
	for chain, generations in dataset.items():
		for gen_i, (subject_a, subject_b) in enumerate(generations):
			full_correct = []
			part_correct = []
			for trial in subject_a['responses']:
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
			axis.plot(x, y, color=chain_name_colors[chain])
	axis.set_ylim(-0.05, 1.05)
	axis.set_xlim(1, 36)
	axis.set_xticks([1, 12, 24, 36])
	axis.set_xlabel('Mini-test trial')
	axis.set_ylabel(f'Mean accuracy over previous {window} trials')
	axis.axvline(12, color='black', linestyle='--')
	axis.axvline(24, color='black', linestyle='--')
	plt.show()



dataset = json_load(ROOT / 'data' / 'pilot3.json')

# plot_response_time(dataset)
# plot_transmission_error(dataset)
# plot_expressivity(dataset)
# plot_informativeness(dataset)
plot_simplicity(dataset)
# plot_simplicity_informativeness(dataset)
# plot_structure(dataset)
# baseline_structure()
# print_word_chains(dataset)
# print_comments(dataset)
# draw_matrices(dataset)
# typ_dist(dataset)
# training_curve(dataset)
