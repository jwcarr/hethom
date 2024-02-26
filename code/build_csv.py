from pathlib import Path
from collections import defaultdict
from itertools import product
import numpy as np
import pandas as pd
import Levenshtein
from utils import json_load
import voi
import matrix


ROOT = Path(__file__).parent.parent.resolve()


ref_systems = [
	matrix.reference_systems['holistic'],
	matrix.reference_systems['expressive'],
	matrix.reference_systems['redundant'],
	matrix.reference_systems['transparent'],
]


def convert_lexicon_meanings_to_tuple(lexicon):
	converted_lexicon = {}
	for item, signal in lexicon.items():
		meaning = tuple(map(int, item.split('_')))
		converted_lexicon[meaning] = signal
	return converted_lexicon

def typological_classification(lexicon):
	system = matrix.make_matrix(lexicon, 3, 3)
	distances = [
		voi.variation_of_information(system, ref_system) for ref_system in ref_systems
	]
	classification = np.argmin(distances)
	return ['H', 'E', 'R', 'D'][classification]

def communicative_cost(lexicon, dims):
	reverse_lexicon = defaultdict(set)
	for meaning, signal in lexicon.items():
		reverse_lexicon[signal].add(meaning)
	U = product(*[range(n_values) for n_values in dims])
	U_size = np.prod(dims)
	return 1 / U_size * sum([-np.log2(1 / len(reverse_lexicon[lexicon[m]])) for m in U])

def transmission_error(lexicon1, lexicon2):
	sum_lev_dist = 0.0
	for item, word1 in lexicon1.items():
		word2 = lexicon2[item]
		sum_lev_dist += Levenshtein.distance(word1, word2)
	return sum_lev_dist / len(lexicon1)

def communicative_success(subject_a_id, subject_b_id):
	subject_a = json_load(ROOT / 'data' / 'exp' / f'subject_{subject_a_id}.json')
	subject_b = json_load(ROOT / 'data' / 'exp' / f'subject_{subject_b_id}.json')
	n_correct = 0
	n_trials = 0
	for a_trial, b_trial in zip(subject_a['responses'], subject_b['responses']):
		if a_trial['test_type'] == 'mini_test':
			continue
		if a_trial['test_type'] == 'comm_production':
			assert b_trial['test_type'] == 'comm_comprehension'
			correct_response = a_trial['item']
			selected_response = b_trial['selected_item']
		if a_trial['test_type'] == 'comm_comprehension':
			assert b_trial['test_type'] == 'comm_production'
			correct_response = b_trial['item']
			selected_response = a_trial['selected_item']
		n_correct += correct_response == selected_response
		n_trials += 1
	return n_correct / n_trials

def build_csv(exp_data_file, exp_csv_file):
	exp_data = json_load(exp_data_file)
	table = []
	for condition, data in exp_data.items():
		print(condition.upper())
		for chain_i, chain in enumerate(data):
			print('  Chain', chain_i)
			prev_lexicon = None
			for generation_i, (subject_a, subject_b) in enumerate(chain):
				if condition.startswith('con') and generation_i > 0:
					epoch_i = ((generation_i - 1) // 3) + 1
					generation_in_epoch = ((generation_i - 1) % 3) + 1
				else:
					epoch_i = 0
					generation_in_epoch = 0
				print('    Generation', generation_i)
				lexicon_a = convert_lexicon_meanings_to_tuple(subject_a['lexicon'])
				classification = typological_classification(lexicon_a)
				cost = communicative_cost(lexicon_a, (3, 3))
				if prev_lexicon:
					error = transmission_error(lexicon_a, prev_lexicon)
				else:
					error = None
				if 'com' in condition and generation_i > 0:
					success = communicative_success(subject_a['subject_id'], subject_b['subject_id'])
				else:
					success = None
				table.append([
					condition,
					chain_i,
					generation_i,
					epoch_i,
					generation_in_epoch,
					classification,
					int(classification in ['H', 'E']),
					cost,
					error,
					success,
				])
				prev_lexicon = lexicon_a
	df = pd.DataFrame(table, columns=['condition', 'chain', 'generation', 'epoch', 'generation_in_epoch', 'type', 'informative', 'cost', 'error', 'success'])
	df.to_csv(exp_csv_file, index=False)


if __name__ == '__main__':

	exp_json_file = ROOT / 'data' / 'exp.json'
	exp_csv_file = ROOT / 'data' / 'exp.csv'

	build_csv(exp_json_file, exp_csv_file)
