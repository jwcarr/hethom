from pathlib import Path
from utils import json_load, json_save


ROOT = Path(__file__).parent.parent.resolve()


def identical_dicts(dict1, dict2):
	for meaning, word1 in dict1.items():
		word2 = dict2[meaning]
		if word1 != word2:
			return False
	return True


def process_subject(exp_id, subject_id):
	if subject_id is None:
		return None, None
	subject_data_file = ROOT / 'data' / exp_id / f'{subject_id}.json'
	subject_data = json_load(subject_data_file)
	return subject_data['input_lexicon'], subject_data['lexicon']


def process_chain(exp_id, chain_id):
	chain_dataset = []
	chain_data_file = ROOT / 'data' / exp_id / f'{chain_id}.json'
	chain_data = json_load(chain_data_file)
	assert len(chain_data['subjects']) == chain_data['current_gen']
	if len(chain_data['subjects']) > 20:
		chain_data['subjects'] = chain_data['subjects'][:20]
	for generation_i, (subject_a, subject_b) in enumerate(chain_data['subjects']):
		input_lexicon_a, output_lexicon_a = process_subject(exp_id, subject_a)
		input_lexicon_b, output_lexicon_b = process_subject(exp_id, subject_b)
		if generation_i == 0:
			generation = [
				{'subject_id': None, 'lexicon': input_lexicon_a},
				None
			]
			chain_dataset.append(generation)
		else:
			assert identical_dicts(input_lexicon_a, chain_dataset[-1][0]['lexicon'])
			if input_lexicon_b:
				assert identical_dicts(input_lexicon_b, chain_dataset[-1][0]['lexicon'])
		if input_lexicon_b:
			assert identical_dicts(input_lexicon_a, input_lexicon_b)
		if subject_b:
			generation = [
				{'subject_id': subject_a, 'lexicon': output_lexicon_a},
				{'subject_id': subject_b, 'lexicon': output_lexicon_b}
			]
		else:
			generation = [
				{'subject_id': subject_a, 'lexicon': output_lexicon_a},
				None
			]
		chain_dataset.append(generation)
	return chain_dataset


def process_experiment(exp_id):
	exp_dataset = {}
	exp_config_file = ROOT / 'experiment' / 'config' / f'{exp_id}.json'
	exp_config = json_load(exp_config_file)
	for task in exp_config['tasks']:
		task_id = task['task_id']
		n_chains = task['n_chains']
		exp_dataset[task_id] = []
		for chain_i in range(n_chains):
			chain_id = f'chain_{task_id}_{chain_i}'
			chain_dataset = process_chain(exp_id, chain_id)
			exp_dataset[task_id].append(chain_dataset)
	output_path = ROOT / 'data' / f'{exp_id}.json'
	json_save(exp_dataset, output_path)


if __name__ == '__main__':

	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('exp_id', action='store', help='Experiment ID')
	args = parser.parse_args()
	
	process_experiment(args.exp_id)
