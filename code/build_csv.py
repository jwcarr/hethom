from pathlib import Path
from collections import defaultdict
from itertools import product
import numpy as np
import pandas as pd
import Levenshtein
from utils import json_load
import mantel


ROOT = Path(__file__).parent.parent.resolve()


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

def communicative_cost(lexicon, dims):
	reverse_lexicon = defaultdict(set)
	for meaning, signal in lexicon.items():
		reverse_lexicon[signal].add(meaning)
	U = product(*[range(n_values) for n_values in dims])
	U_size = np.product(dims)
	return 1 / U_size * sum([-np.log2(1 / len(reverse_lexicon[lexicon[m]])) for m in U])

def structure(lexicon, dims):
	strings = []
	meanings = []
	for meaning, signal in lexicon.items():
		strings.append(signal)
		meanings.append(meaning)
	string_dists = []
	for i in range(len(strings)):
		for j in range(i+1, len(strings)):
			string_dists.append(Levenshtein.distance(strings[i], strings[j]))
	meaning_dists = []
	for i in range(len(meanings)):
		for j in range(i+1, len(meanings)):
			meaning_dists.append(Levenshtein.distance(meanings[i], meanings[j]))
	return mantel.test(string_dists, meaning_dists).z

def communicative_success(subject_a_id, subject_b_id):
	subject_a = json_load(ROOT / 'data' / 'exp' / f'subject_{subject_a_id}.json')
	subject_b = json_load(ROOT / 'data' / 'exp' / f'subject_{subject_b_id}.json')

	n_correct = 0
	for response_a, response_b in zip(subject_a['responses'], subject_b['responses']):
		if response_a['test_type'] == 'mini_test':
			continue
		assert response_a['item'] == response_b['item']
		if response_a['test_type'] == 'comm_production':
			n_correct += response_a['item'] == response_b['selected_item']
		else:
			n_correct += response_b['item'] == response_a['selected_item']

	return n_correct

def build_csv(exp_data_file, exp_csv_file):
	exp_data = json_load(exp_data_file)
	table = []
	for condition, data in exp_data.items():
		print(condition.upper())
		for chain_i, chain in enumerate(data):
			print('  Chain', chain_i)
			prev_lexicon = None
			for generation_i, (subject_a, subject_b) in enumerate(chain):
				if condition.startswith('con'):
					if generation_i == 0:
						epoch_i = 0
					elif generation_i < 4:
						epoch_i = 1
					elif generation_i < 7:
						epoch_i = 2
					else:
						epoch_i = 3
				else:
					epoch_i = 0
				print('    Generation', generation_i)
				lexicon_a = convert_lexicon_meanings_to_tuple(subject_a['lexicon'])
				error = transmission_error(lexicon_a, prev_lexicon) if prev_lexicon else None
				cost = communicative_cost(lexicon_a, (3, 3))
				struc = structure(lexicon_a, (3, 3))
				if subject_b:
					lexicon_b = convert_lexicon_meanings_to_tuple(subject_b['lexicon'])
					success = communicative_success(subject_a['subject_id'], subject_b['subject_id'])
				else:
					success = None
				table.append([
					condition,
					chain_i,
					generation_i,
					epoch_i,
					cost,
					error,
					success,
					struc,
				])
				prev_lexicon = lexicon_a
	df = pd.DataFrame(table, columns=['condition', 'chain', 'generation', 'epoch', 'cost', 'error', 'comm_success', 'structure'])
	df.to_csv(exp_csv_file, index=False)


if __name__ == '__main__':

	exp_json_file = ROOT / 'data' / 'exp.json'
	exp_csv_file = ROOT / 'data' / 'exp.csv'

	build_csv(exp_json_file, exp_csv_file)
