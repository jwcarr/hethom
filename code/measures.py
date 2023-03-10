from pathlib import Path
import json
import matplotlib.pyplot as plt


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

dataset = json_load(ROOT / 'data' / 'pilot1.json')


for chain, generations in dataset.items():
	error = []
	exprs = []
	for generation in generations:
		error.append(
			transmission_error(generation['input_lexicon'], generation['lexicon'])
		)
		exprs.append(
			expressivity(generation['lexicon'])
		)
	# plt.plot(range(1, len(error) + 1), error)
	plt.plot(range(1, len(error) + 1), exprs)
plt.show()