'''

This code is used to monitor the online experiments by interacting with the
MongoDB database. It can be used from the command line, e.g.:

	python mission_control.py some_task_id --status

of from a Python console:

	import mission_control as mc
	mc.status('some_task_id')

If the database is on a remote server, you can either run this module on the
remote or create an SSH tunnel, so that, e.g., port 27017 on your local
machine is mapped to port 27017 on the remote, e.g.:

	ssh -L 27017:localhost:27017 jon@joncarr.net

The general workflow is:

- launch: launch a task (as defined by the parameters in a JSON file)
- status: monitor the status of a task
- start/stop: make a task active/inactive
- exclude: exclude a dropout participant and increment the slots on the task
- approve/bonus: print an approval or bonusing list to submit to Prolific
- pull: pull down the data from all completed participants on a task

'''

import json
import random
import time
import numpy as np
from itertools import product
from pathlib import Path
from pymongo import MongoClient
from bson.json_util import dumps


CONFIG_DIR = Path('config')
DATA_DIR = Path('../data')
DOMAIN = 'localhost'
PORT = 27017


db = MongoClient(DOMAIN, PORT)


def choose(choices, choose_n):
	'''
	Choose N items from choices without replacement.
	'''
	assert choose_n <= len(choices)
	random.shuffle(choices)
	return choices[:choose_n]

def create_seed_lexicon(task):
	'''
	Use the task definition to create the generation-0 seed words and their
	mapping onto the objects.
	'''
	assert task['n_shapes'] == len(task['stems'])
	assert len(task['seed_suffix_spellings']) == len(task['seed_suffix_variation'])
	for grapheme_set, n_variants in zip(task['seed_suffix_spellings'], task['seed_suffix_variation']):
		assert n_variants <= len(grapheme_set)
	seed_suffix_spellings = [
		choose(grapheme_set, choose_n)
		for grapheme_set, choose_n in zip(task['seed_suffix_spellings'], task['seed_suffix_variation'])
	]
	endings = [
		''.join(grapheme_combination)
		for grapheme_combination in product(*seed_suffix_spellings)
	]
	random.shuffle(endings)
	n_required_words = task['n_shapes'] * task['n_colors']
	if len(endings) < n_required_words:
		if n_required_words % len(endings) == 0:
			endings *= n_required_words // len(endings)
		else:
			endings *= (n_required_words // len(endings)) + 1
	endings = endings[:n_required_words]
	random.shuffle(endings)
	object_word_mapping = {}
	for i in range(task['n_shapes']):
		for j in range(task['n_colors']):
			object_word_mapping[f'{i}_{j}'] = task['stems'][i] + endings.pop()
	return object_word_mapping

def launch(exp_id, _=None):
	exp_file = CONFIG_DIR / f'{exp_id}.json'
	with open(exp_file) as file:
		exp = json.load(file)
	assert exp_id == exp['exp_id']
	if exp_id in db.list_database_names():
		raise ValueError('Database already exsits; cannot launch')
	for task in exp['tasks']:
		task['return_url'] = exp['return_url']
		for chain_i in range(exp['chains_per_task']):
			db[exp_id].chains.insert_one({
				'chain_id': f'{task["task_id"]}_{chain_i}',
				'task': task,
				'status': 'available',
				'current_gen': 0,
				'subjects': [],
				'lexicon': create_seed_lexicon(task),
			})
	print('Launched task:', exp['exp_id'])

def erase(exp_id, _=None):
	if input(f'Are you sure you want to completely erase database {exp_id}? ') == 'yes':
		db.drop_database(exp_id)

def status(exp_id, _=None):
	n_available = 0
	for chain in db[exp_id].chains.find():
		print(chain['chain_id'], chain['current_gen'], chain['status'])
		if chain['status'] == 'available':
			n_available += 1
	print('AVAILABLE PLACES', n_available)

def monitor(exp_id, sub_id=None):
	if sub_id is None:
		subjects = db[exp_id].subjects.find({'status': 'active'})
	else:
		subjects = [ db[exp_id].subjects.find_one({'prolific_id': sub_id}) ]
	current_time = int(time.time())
	for subject in subjects:
		print('------------------------------------------')
		print(subject['prolific_id'])
		print('------------------------------------------')
		print('Status:', subject['status'])
		minutes = (current_time - subject['creation_time']) // 60
		seconds = (current_time - subject['creation_time']) % 60
		print('Time since start of experiment:', f'{minutes}:{str(seconds).zfill(2)}')
		minutes = (current_time - subject['modified_time']) // 60
		seconds = (current_time - subject['modified_time']) % 60
		print('Time since last response:', f'{minutes}:{str(seconds).zfill(2)}')
		print('Sequence position:', subject['sequence_position'])
		print('Current event:', subject['trial_sequence'][subject['sequence_position']]['event'])
		print('Client ID:', subject['client_id'])
		print('Reinitializations:', subject['n_reinitializations'])

def entropy(distribution):
	distribution /= distribution.sum()
	return -sum([p * np.log(p) for p in distribution if p > 0])

def review(exp_id, sub_id=None):
	if sub_id is None:
		raise ValueError('Subject ID must be specified')
	subject = db[exp_id].subjects.find_one({'prolific_id': sub_id})
	if subject['status'] != 'approval_needed':
		raise ValueError(f'Cannot review subject with status {subject["status"]}')
	minutes = (subject['modified_time'] - subject['creation_time']) // 60
	seconds = (subject['modified_time'] - subject['creation_time']) % 60
	lexicon = {}
	response_times = []
	attention_checks_passed = 0
	button_distributon = np.zeros(len(subject['input_lexicon']))
	object_distributon = np.zeros(len(subject['input_lexicon']))
	item_to_index_map = sorted(list(subject['input_lexicon'].keys()))
	for trial in subject['responses']:
		if trial['test_type'] == 'mini_test':
			if trial['catch_trial']:
				attention_checks_passed += trial['object_clicked']
			response_times.append(trial['response_time'])
		elif trial['test_type'] == 'test_production':
			trial['item'] = f'{trial["shape"]}_{trial["color"]}'
			lexicon[trial['item']] = trial['input_label']
			response_times.append(trial['response_time'])
		elif trial['test_type'] == 'test_comprehension':
			button_distributon[trial['selected_button']] += 1
			object_distributon[item_to_index_map.index(trial['selected_item'])] += 1
			response_times.append(trial['response_time'])
	lexicon = {item: lexicon[item] for item in sorted(lexicon)}
	for item, word in lexicon.items():
		taught_word = subject['input_lexicon'][item]
		correct = '✅' if word == taught_word else '❌'
		trained = '➡️ ' if item in subject['training_items'] else '  '
		print(item, taught_word.ljust(9, ' '), trained, word.ljust(9, ' '), correct)
	db[exp_id].subjects.update_one({'prolific_id': sub_id}, {'$set':{'status': 'reviewed', 'lexicon': lexicon}})
	print('Time taken:', f'{minutes}:{str(seconds).zfill(2)}')
	print('Mean response time', round(np.mean(response_times) / 1000, 2))
	print('Attention checks:', attention_checks_passed)
	print('Button entropy:', round(entropy(button_distributon), 2))
	print('Object entropy:', round(entropy(object_distributon), 2))
	print('Total bonus:', subject['total_bonus'])
	print('Comments:', subject['comments'])

def convert_to_pounds(int_bonus_in_pence):
	if int_bonus_in_pence >= 100:
		pounds = str(int_bonus_in_pence)[:-2]
		pennys = str(int_bonus_in_pence)[-2:]
	elif int_bonus_in_pence >= 10:
		pounds = '0'
		pennys = str(int_bonus_in_pence)
	else:
		pounds = '0'
		pennys = str(int_bonus_in_pence).zfill(2)
	return f'{pounds}.{pennys}'

def log_approval(subject_id, bonus):
	with open(DATA_DIR / f'{exp_id}_approval_log', 'a') as file:
		file.write(f'{sub_id}\n')
	with open(DATA_DIR / f'{exp_id}_bonus_log', 'a') as file:
		bonus = convert_to_pounds(subject['total_bonus'])
		file.write(f'{sub_id},{bonus}\n')

def approve(exp_id, sub_id=None):
	if sub_id is None:
		raise ValueError('Subject ID must be specified')
	subject = db[exp_id].subjects.find_one({'prolific_id': sub_id})
	if subject['status'] != 'reviewed':
		raise ValueError('Subject not yet reviewed or already approved')
	db[exp_id].subjects.update_one({'prolific_id': sub_id}, {'$set':{'status': 'approved'}})
	chain = db[exp_id].chains.find_one({'chain_id': subject['chain_id']})
	if chain['current_gen'] >= (chain['max_gen'] - 1):
		update_status = 'completed'
	else:
		update_status = 'available'
	db[exp_id].chains.update_one({'chain_id': subject['chain_id']}, {
		'$set': {'status': update_status, 'lexicon': subject['lexicon']},
		'$push': {'subjects': subject['prolific_id']},
		'$inc': {'current_gen': 1},
	})
	log_approval(subject['prolific_id'], subject['total_bonus'])

def reject(exp_id, sub_id=None):
	if sub_id is None:
		raise ValueError('Subject ID must be specified')
	subject = db[exp_id].subjects.find_one({'prolific_id': sub_id})
	if subject['status'] != 'reviewed':
		raise ValueError('Subject not yet reviewed or already approved')
	db[exp_id].subjects.update_one({'prolific_id': sub_id}, {'$set':{'status': 'rejected'}})
	db[exp_id].chains.update_one({'chain_id': subject['chain_id']}, {'$set': {'status': 'available'}})
	log_approval(subject['prolific_id'], subject['total_bonus'])

def drop(exp_id, sub_id=None):
	if sub_id is None:
		raise ValueError('Subject ID must be specified')
	subject = db[exp_id].subjects.find_one({'prolific_id': sub_id})
	if subject['status'] == 'active':
		db[exp_id].subjects.update_one({'prolific_id': sub_id}, {'$set':{'status': 'dropout'}})
		db[exp_id].chains.update_one({'chain_id': subject['chain_id']}, {'$set': {'status': 'available'}})
	else:
		print('Subject not currently active')

def dump(exp_id, _=None):
	subject_id_map = {}
	subject_data = {}
	for i, subject in enumerate(db[exp_id].subjects.find({'status': 'approved'}), 1):
		anon_subject_id = f'{exp_id}_{str(i).zfill(3)}' 
		subject_id_map[ subject['prolific_id'] ] = anon_subject_id
		subject['subject_id'] = anon_subject_id
		del subject['_id']
		del subject['prolific_id']
		file_path = DATA_DIR / exp_id / f'{anon_subject_id}.json'
		with open(file_path, 'w') as file:
			file.write(dumps(subject, indent='\t'))
		subject_data[anon_subject_id] = subject

	dataset = {}
	for chain in db[exp_id].chains.find({}):
		del chain['_id']
		chain['subjects'] = [
			subject_id_map[prolific_id] for prolific_id in chain['subjects']
		]
		file_path = DATA_DIR / exp_id / f'chain_{chain["chain_id"]}.json'
		with open(file_path, 'w') as file:
			file.write(dumps(chain, indent='\t'))
		dataset[ chain['chain_id'] ] = [ subject_data[subject_id] for subject_id in chain['subjects'] ]
	with open(DATA_DIR / f'{exp_id}.json', 'w') as file:
		file.write(dumps(dataset), indent='\t')


if __name__ == '__main__':

	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('exp_id', action='store', help='Experiment ID')
	parser.add_argument('action', action='store', help='Action to take')
	parser.add_argument('sub_id', action='store', nargs='?', default=None, help='Subject ID')
	args = parser.parse_args()

	{
		'launch': launch,
		'erase': erase,
		'status': status,
		'monitor': monitor,
		'review': review,
		'approve': approve,
		'drop': drop,
		'reject': reject,
		'dump': dump,
	}[args.action](args.exp_id, args.sub_id)
