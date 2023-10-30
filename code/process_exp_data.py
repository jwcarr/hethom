from pathlib import Path
from utils import json_load, json_save


ROOT = Path(__file__).parent.parent.resolve()
DATA_PATH = ROOT / 'data'


ABRIDGMENT_ITEMS = ['subject_id', 'lexicon', 'spoken_forms', 'training_items']


def identical_dicts(d1, d2):
	for k, v in d1.items():
		if d2[k] != v:
			return False
	return True


def subject_a_is_dominant(subject_a, subject_b, test_alternative=True):
	'''
	Check to see whether subject A's trial order is dominant over subject B's.
	The dominant subject is the one whose trial order has unseen production
	tests before unseen comprehension tests. Usually this is subject A but it
	can be subject B in cases where the original subject A was replaced.
	'''
	if subject_b is None:
		return True
	subject_a_prod = [trial['payload']['item'] for trial in subject_a['trial_sequence'] if trial['event'] == 'comm_production']
	subject_a_comp = [trial['payload']['item'] for trial in subject_b['trial_sequence'] if trial['event'] == 'comm_production']
	for production_index, item in enumerate(subject_a_prod):
		if item not in subject_a['training_items'] and subject_a_comp.index(item) < production_index:
			if test_alternative and subject_a_is_dominant(subject_b, subject_a, test_alternative=False) == False:
				print('‼️ WARNING: Neither subject is dominant. Iterating subject A.')
				return True
			return False
	return True


def individual_check(exp_id, n_subjects):
	'''
	This runs a series of checks on each individual subject's raw output data.
	'''
	for i in range(1, n_subjects + 1):
		subject_id = str(i).zfill(3)
		data = json_load(DATA_PATH / exp_id / f'subject_{subject_id}.json')

		assert data['client_id'] is None
		assert data['modified_time'] > data['creation_time']
		assert data['status'] in ['approved', 'rejected', 'jilted']
		assert len(data['chain_id']) == 9
		assert data['generation'] <= 9
		assert len(data['input_lexicon']) == 9
		assert len(data['spoken_forms']) == 9
		assert len(data['training_items']) == 6
		assert len(data['trial_sequence']) == 60
		assert data['comments'] is None

		if 'dif' in data['chain_id']:
			assert len(set(data['spoken_forms'].values())) == 3
		
		elif 'con' in data['chain_id']:
			if data['generation'] > 0 and data['generation'] < 4:
				assert len(set(data['spoken_forms'].values())) == 9
			elif data['generation'] > 3 and data['generation'] < 7:
				assert len(set(data['spoken_forms'].values())) == 6
			elif data['generation'] > 6 and data['generation'] < 10:
				assert len(set(data['spoken_forms'].values())) == 3

		prev_time = data['creation_time']
		for trial in data['responses']:
			assert trial['time'] > prev_time
			assert int(trial['time'] - trial['response_time']/1000) >= (prev_time - 1)
			prev_time = trial['time']

		prev_prog = -1
		for trial in data['trial_sequence'][1:]:
			assert trial['payload']['progress'] > prev_prog
			prev_prog = trial['payload']['progress']

		mt_req = [trial['payload']['test_trial'] for trial in data['trial_sequence'] if trial['event'] == 'training_block']
		mt_res = [trial for trial in data['responses'] if trial['test_type'] == 'mini_test']

		assert len(mt_req) == 36
		assert set(data['training_items']) == set([trial['item'] for trial in mt_req])
		if data['status'] == 'approved' or data['status'] == 'rejected':
			assert len(mt_req) == len(mt_res) == 36

		bonus = 0
		checks_passed = 0
		for req, res in zip(mt_req, mt_res):
			assert req['catch_trial'] == res['catch_trial']
			assert req['shape'] == res['shape']
			assert req['color'] == res['color']
			assert req['word'] == res['expected_label']
			if req['word'] == res['input_label'] and res['response_time'] < 20000:
				bonus += 2
			if req['catch_trial']:
				checks_passed += res['object_clicked']
		if data['status'] == 'approved':
			assert checks_passed >= 2

		if 'lrn' in data['chain_id']:

			tp_req = [trial['payload'] for trial in data['trial_sequence'] if trial['event'] == 'test_production']
			tp_res = [trial for trial in data['responses'] if trial['test_type'] == 'test_production']
		
			assert len(tp_req) == 9
			if data['status'] != 'jilted':
				assert len(tp_res) == 9
			for req, res in zip(tp_req, tp_res):
				assert req['shape'] == res['shape']
				assert req['color'] == res['color']
				assert req['word'] == res['expected_label']
				if req['word'] == res['input_label']:
					bonus += 2

			tc_req = [trial['payload'] for trial in data['trial_sequence'] if trial['event'] == 'test_comprehension']
			tc_res = [trial for trial in data['responses'] if trial['test_type'] == 'test_comprehension']

			assert len(tc_req) == 9
			if data['status'] != 'jilted':
				assert len(tc_res) == 9
			for req, res in zip(tc_req, tc_res):
				assert req['word'] == res['word']
				assert req['array'][res['selected_button']] == res['selected_item']
				if res['selected_item'] in req['items']:
					bonus += 2
			assert bonus == data['total_bonus']

			lexicon = {f'{trial["shape"]}_{trial["color"]}' : trial['input_label'] for trial in tp_res}

		if 'com' in data['chain_id']:

			cp_req = [trial['payload'] for trial in data['trial_sequence'] if trial['event'] == 'comm_production']
			cp_res = [trial for trial in data['responses'] if trial['test_type'] == 'comm_production']

			assert len(cp_req) == 9
			if data['status'] != 'jilted':
				assert len(cp_res) == 9
			for req, res in zip(cp_req, cp_res):
				assert req['item'] == res['item']
				assert req['shape'] == res['shape']
				assert req['color'] == res['color']
				assert req['word'] == res['expected_label']

			cc_req = [trial['payload'] for trial in data['trial_sequence'] if trial['event'] == 'comm_comprehension']
			cc_res = [trial for trial in data['responses'] if trial['test_type'] == 'comm_comprehension']

			assert len(cc_req) == 9
			if data['status'] != 'jilted':
				assert len(cc_res) == 9
			for req, res in zip(cc_req, cc_res):
				assert req['array'][res['selected_button']] == res['selected_item']

			lexicon = {f'{trial["shape"]}_{trial["color"]}' : trial['input_label'] for trial in cp_res}


		if data['status'] == 'approved' or data['status'] == 'rejected':
			assert len(data['lexicon']) == 9
			assert identical_dicts(data['lexicon'], lexicon)

		if data['status'] == 'rejected':
			assert 'reason_for_rejection' in data

		assert data['total_bonus'] <= 108
		print(f'Subject {subject_id} passed with no errors')


def iteration_check(exp_id, conditions, n_chains):
	'''
	This checks for potential errors across iterated learning chains.
	'''
	for condition in conditions:
		for chain_i in range(n_chains):
			chain_data = json_load(DATA_PATH / exp_id / f'chain_{condition}_{chain_i}.json')

			assert chain_data['status'] == 'completed'
			assert chain_data['current_gen'] == 9
			assert chain_data['subject_a'] == chain_data['subject_b'] == None
			assert len(chain_data['subjects']) == 9
			assert len(chain_data['lexicon']) == 9
			if 'dif' in chain_data['chain_id']:
				assert chain_data['sound_epoch'] == 0
				assert len(chain_data['spoken_forms']) == 1
				assert len(set(chain_data['spoken_forms'][0].values())) == 3
			elif 'con' in chain_data['chain_id']:
				assert chain_data['sound_epoch'] == 3
				assert len(chain_data['spoken_forms']) == 3
				assert len(set(chain_data['spoken_forms'][0].values())) == 9
				assert len(set(chain_data['spoken_forms'][1].values())) == 6
				assert len(set(chain_data['spoken_forms'][2].values())) == 3

			if 'lrn' in chain_data['chain_id']:

				prev_lexicon = None
				for gen, (subject_a, _) in enumerate(chain_data['subjects'], 1):
					assert _ is None
					a_data = json_load(DATA_PATH / exp_id / f'subject_{subject_a}.json')
					assert a_data['status'] == 'approved'
					assert a_data['chain_id'] == chain_data['chain_id']
					assert a_data['generation'] == gen
					if prev_lexicon is None:
						prev_lexicon = a_data['input_lexicon']
					assert identical_dicts(prev_lexicon, a_data['input_lexicon'])
					prev_lexicon = a_data['lexicon']
				assert identical_dicts(prev_lexicon, chain_data['lexicon'])

			elif 'com' in chain_data['chain_id']:

				prev_lexicon = None
				for gen, (subject_a, subject_b) in enumerate(chain_data['subjects'], 1):
					a_data = json_load(DATA_PATH / exp_id / f'subject_{subject_a}.json')
					b_data = json_load(DATA_PATH / exp_id / f'subject_{subject_b}.json')
					assert a_data['status'] == b_data['status'] == 'approved'
					assert a_data['chain_id'] == b_data['chain_id'] == chain_data['chain_id']
					assert a_data['generation'] == b_data['generation'] == gen
					if prev_lexicon is None:
						prev_lexicon = a_data['input_lexicon']
					assert identical_dicts(prev_lexicon, a_data['input_lexicon'])
					assert identical_dicts(prev_lexicon, b_data['input_lexicon'])
					assert identical_dicts(a_data['input_lexicon'], b_data['input_lexicon'])
					if subject_a_is_dominant(a_data, b_data):
						prev_lexicon = a_data['lexicon']
					else:
						prev_lexicon = b_data['lexicon']
						print(f'-  In {b_data["chain_id"]}, generation {b_data["generation"]}, subject B ({b_data["subject_id"]}) was iterated instead of subject A ({a_data["subject_id"]}) because the original subject A dropped out.')
				assert identical_dicts(prev_lexicon, chain_data['lexicon'])
			print(f'Chain {chain_data["chain_id"]} passed with no errors')


def communication_check(exp_id, conditions, n_chains):
	'''
	This checks for potential errors across communicating partners.
	'''
	for condition in conditions:
		for chain_i in range(n_chains):
			chain_data = json_load(DATA_PATH / exp_id / f'chain_{condition}_{chain_i}.json')
			for gen, (subject_a, subject_b) in enumerate(chain_data['subjects'], 1):
				a_data = json_load(DATA_PATH / exp_id / f'subject_{subject_a}.json')
				b_data = json_load(DATA_PATH / exp_id / f'subject_{subject_b}.json')

				a_mt_bonus = 0
				b_mt_bonus = 0
				comm_bonus = 0
				for a_response, b_response in zip(a_data['responses'], b_data['responses']):
					if a_response['test_type'] == 'mini_test':
						assert a_response['test_type'] == b_response['test_type']
						if a_response['expected_label'] == a_response['input_label'] and a_response['response_time'] < 20000:
							a_mt_bonus += 2
						if b_response['expected_label'] == b_response['input_label'] and b_response['response_time'] < 20000:
							b_mt_bonus += 2
					else:
						assert a_response['item'] == b_response['item']
						if a_response['test_type'] == 'comm_production':
							assert b_response['test_type'] == 'comm_comprehension'
							assert a_response['input_label'] == b_response['word']
							if a_response['item'] == b_response['selected_item']:
								comm_bonus += 2
						elif a_response['test_type'] == 'comm_comprehension':
							assert b_response['test_type'] == 'comm_production'
							assert b_response['input_label'] == a_response['word']
							if b_response['item'] == a_response['selected_item']:
								comm_bonus += 2
				assert a_mt_bonus + comm_bonus == a_data['total_bonus']
				assert b_mt_bonus + comm_bonus == b_data['total_bonus']
			print(f'Chain {chain_data["chain_id"]} passed with no errors')


def create_abridged_data_file(exp_id, conditions, n_chains):
	'''
	Creates a single-file, short-version of the entire experimental dataset
	that is easier to handle.
	'''
	exp_dataset = {condition: [] for condition in conditions}
	for condition in conditions:
		for chain_i in range(n_chains):
			chain_id = f'{condition}_{chain_i}'
			chain_dataset = []
			chain_data = json_load(DATA_PATH / exp_id / f'chain_{chain_id}.json')
			for generation_i, (subject_a, subject_b) in enumerate(chain_data['subjects']):
				data_a = json_load(DATA_PATH / exp_id / f'subject_{subject_a}.json')
				data_b = json_load(DATA_PATH / exp_id / f'subject_{subject_b}.json') if subject_b is not None else None
				if generation_i == 0:
					chain_dataset.append([
						{'subject_id': None, 'lexicon': data_a['input_lexicon']},
						None
					])
				if data_b and not subject_a_is_dominant(data_a, data_b):
					data_a, data_b = data_b, data_a
				assert identical_dicts(data_a['input_lexicon'], chain_dataset[-1][0]['lexicon'])
				assert identical_dicts(data_b['input_lexicon'], chain_dataset[-1][0]['lexicon']) if data_b else True
				data_a_abridged = {key: data_a[key] for key in ABRIDGMENT_ITEMS}
				data_b_abridged = {key: data_b[key] for key in ABRIDGMENT_ITEMS} if data_b is not None else None
				chain_dataset.append([data_a_abridged, data_b_abridged])
			exp_dataset[condition].append(chain_dataset)
	json_save(exp_dataset, DATA_PATH / f'{exp_id}.json')


if __name__ == '__main__':

	individual_check('exp', 584)

	iteration_check('exp', ['dif_lrn', 'dif_com', 'con_lrn', 'con_com'], 10)

	communication_check('exp', ['dif_com', 'con_com'], 10)

	create_abridged_data_file('exp', ['dif_lrn', 'dif_com', 'con_lrn', 'con_com'], 10)
