'''

This code is used to monitor the online experiments by interacting with the
MongoDB database. It can be used from the command line, e.g.:

	python mission_control.py some_task_id status

of from a Python console:

	import mission_control as mc
	exp = mc.MissionControl('some_task_id')
	exp.status()

If the database is on a remote server, you can either run this module on the
remote or create an SSH tunnel, so that, e.g., port 27017 on your local
machine is mapped to port 27017 on the remote, e.g.:

	ssh -L 27017:localhost:27017 jon@joncarr.net

The general workflow is:

- launch: launch a task (as defined by the parameters in a JSON config file)
- status: monitor the status of the chains
- monitor: view current subject activity
- open/close: open or close a chain
- review: review the outcome of a chain that is awaiting approval
- approve: approve the outcome and reopen the chain for a new generation
- dump: pull down the data from all completed participants on a task

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

DB = MongoClient(DOMAIN, PORT)

STATUS_EMOJI = {
	'available': '🟢',
	'unavailable': '🔴',
	'converged': '⚫️',
	'closed': '🟡',
	'approval_needed': '🔵',
}


class MissionControl:

	def __init__(self, exp_id):
		self.exp_id = exp_id
		if self.exp_id in DB.list_database_names():
			self.db = DB[self.exp_id]
		else:
			self.db = None

	def _log_approval(self, sub_id, bonus):
		with open(DATA_DIR / f'{self.exp_id}_approval_log', 'a') as file:
			file.write(f'{sub_id}\n')
		with open(DATA_DIR / f'{self.exp_id}_bonus_log', 'a') as file:
			bonus = convert_to_pounds(bonus)
			file.write(f'{sub_id},{bonus}\n')

	def launch(self, _=None):
		exp_config_file = CONFIG_DIR / f'{self.exp_id}.json'
		with open(exp_config_file) as file:
			exp_config = json.load(file)
		assert self.exp_id == exp_config['exp_id']
		if self.exp_id in DB.list_database_names():
			raise ValueError('Database already exsits; cannot launch')
		for task in exp_config['tasks']:
			task['return_url'] = exp_config['return_url']
			for chain_i in range(task['n_chains']):
				if 'lexicon' in task:
					lexicon = task['lexicon']
				else:
					lexicon = create_seed_lexicon(task)
				DB[self.exp_id].chains.insert_one({
					'chain_id': f'{task["task_id"]}_{chain_i}',
					'task': task,
					'status': 'closed',
					'current_gen': 0,
					'subject_a': None,
					'subject_b': None,
					'subjects': [],
					'lexicon': lexicon,
				})
		print('Launched task:', self.exp_id)

	def erase(self, _=None):
		if input(f'Are you sure you want to completely erase database "{self.exp_id}"? ') == 'yes':
			DB.drop_database(self.exp_id)
			self.db = None

	def status(self, _=None):
		n_available = 0
		for chain in self.db.chains.find():
			if chain['task']['communication']:
				print(chain['chain_id'], str(chain['current_gen']).rjust(2, ' '), STATUS_EMOJI[chain['status']], chain['status'], chain['subject_a'], chain['subject_b'])
			else:
				print(chain['chain_id'], str(chain['current_gen']).rjust(2, ' '), STATUS_EMOJI[chain['status']], chain['status'], chain['subject_a'])
			if chain['status'] == 'available':
				if chain['task']['communication']:
					n_available += 2
				else:
					n_available += 1
		print('AVAILABLE PLACES', n_available)

	def monitor(self, _=None):
		subjects = self.db.subjects.find({'status': 'active'}).sort('chain_id', 1)
		current_time = int(time.time())
		for subject in subjects:
			print('------------------------------------------')
			print(subject['subject_id'], subject['chain_id'])
			print('------------------------------------------')
			print('Status:', subject['status'])
			minutes = (current_time - subject['creation_time']) // 60
			seconds = (current_time - subject['creation_time']) % 60
			print('Time since start of experiment:', f'{minutes}:{str(seconds).zfill(2)}')
			minutes = (current_time - subject['modified_time']) // 60
			seconds = (current_time - subject['modified_time']) % 60
			long_time_alert = '‼️' if minutes >= 1 else ''
			print('Time since last response:', f'{minutes}:{str(seconds).zfill(2)}', long_time_alert)
			print('Sequence position:', subject['sequence_position'])
			print('Current event:', subject['trial_sequence'][subject['sequence_position']]['event'])
			disconnect_alert = '‼️' if subject['client_id'] is None else ''
			print('Client ID:', subject['client_id'], disconnect_alert)
			print('Reinitializations:', subject['n_reinitializations'])
		print('############################################################################')

	def dump(self, _=None, approved_IDs=None):
		subject_id_map = {None: None}
		subject_data = {None: None}
		exp_dir = DATA_DIR / self.exp_id
		if not exp_dir.exists():
			exp_dir.mkdir()
		if approved_IDs is None:
			find = {'status': 'approved'}
		else:
			find = {'subject_id': {'$in': approved_IDs}}
		for i, subject in enumerate(self.db.subjects.find(find), 1):
			del subject['_id']
			subject['client_id'] = None
			anon_subject_id = f'{self.exp_id}_{str(i).zfill(3)}'
			subject_id_map[ subject['subject_id'] ] = anon_subject_id
			subject['subject_id'] = anon_subject_id
			with open(exp_dir / f'{anon_subject_id}.json', 'w') as file:
				file.write(dumps(subject, indent='\t'))
			subject_data[anon_subject_id] = subject
		# dataset = {}
		for chain in self.db.chains.find({}):
			del chain['_id']
			chain['subjects'] = [
				(subject_id_map[subject_a], subject_id_map[subject_b]) for subject_a, subject_b in chain['subjects']
			]
			with open(exp_dir / f'chain_{chain["chain_id"]}.json', 'w') as file:
				file.write(dumps(chain, indent='\t'))
		# 	dataset[ chain['chain_id'] ] = [
		# 		(subject_data[subject_a], subject_data[subject_b]) for subject_a, subject_b in chain['subjects']
		# 	]
		# with open(DATA_DIR / f'{self.exp_id}.json', 'w') as file:
		# 	file.write(dumps(dataset, indent='\t'))

	def open(self, chain_id=None):
		if chain_id is None:
			raise ValueError('Chain ID must be specified')
		chain = self.db.chains.find_one({'chain_id': chain_id})
		if chain['status'] != 'closed':
			raise ValueError('Cannot open: Chain not currently closed')
		self.db.chains.update_one({'chain_id': chain['chain_id']}, {'$set': {'status': 'available'}})

	def close(self, chain_id=None):
		if chain_id is None:
			raise ValueError('Chain ID must be specified')
		chain = self.db.chains.find_one({'chain_id': chain_id})
		if chain['subject_a'] is None and chain['subject_b'] is None and chain['status'] == 'available':
			self.db.chains.update_one({'chain_id': chain['chain_id']}, {'$set': {'status': 'closed'}})
		else:
			raise ValueError('Cannot close: Chain occupied')

	def review(self, chain_id=None):
		if chain_id is None:
			raise ValueError('Chain ID must be specified')
		chain = self.db.chains.find_one({'chain_id': chain_id})
		if chain is None:
			raise ValueError('Chain not found')
		if chain['status'] != 'approval_needed':
			raise ValueError('Chain not awaiting approval')
		self.review_subject(chain['subject_a'])
		self.review_subject(chain['subject_b'])

	def approve(self, chain_id=None, do_not_reopen=False):
		if chain_id is None:
			raise ValueError('Chain ID must be specified')
		chain = self.db.chains.find_one({'chain_id': chain_id})
		if chain is None:
			raise ValueError('Chain not found')
		if chain['status'] != 'approval_needed':
			raise ValueError('Chain not awaiting approval')
		subject_a = self.approve_subject(chain['subject_a'])
		subject_b = self.approve_subject(chain['subject_b'])
		chain_converged = False
		for item, word in chain['lexicon'].items():
			if subject_a['lexicon'][item] != word:
				break
		else: # for loop exits normally, all words match, chain has converged
			chain_converged = True
		if chain_converged:
			update_status = 'converged'
		else:
			if do_not_reopen:
				update_status = 'closed'
			else:
				update_status = 'available'
		self.db.chains.update_one({'chain_id': chain['chain_id']}, {
			'$set': {'status': update_status, 'lexicon': subject_a['lexicon'], 'subject_a': None, 'subject_b': None},
			'$push': {'subjects': [chain['subject_a'], chain['subject_b']]},
			'$inc': {'current_gen': 1},
		})

	def reject(self, chain_id=None, do_not_reopen=False):
		if chain_id is None:
			raise ValueError('Chain ID must be specified')
		chain = self.db.chains.find_one({'chain_id': chain_id})
		if chain is None:
			raise ValueError('Chain not found')
		if chain['status'] != 'approval_needed':
			raise ValueError('Chain not awaiting approval')
		self.reject_subject(chain['subject_a'])
		self.reject_subject(chain['subject_b'])
		if do_not_reopen:
			update_status = 'closed'
		else:
			update_status = 'available'
		self.db.chains.update_one({'chain_id': chain['chain_id']}, {'$set': {'status': update_status, 'subject_a': None, 'subject_b': None}})

	def drop(self, chain_id=None, do_not_reopen=True):
		'''
		Set both subjects' statuses to dropout and reset the chain.
		'''
		if chain_id is None:
			raise ValueError('Chain ID must be specified')
		chain = self.db.chains.find_one({'chain_id': chain_id})
		if chain is None:
			raise ValueError('Chain not found')
		if chain['status'] != 'unavailable':
			raise ValueError(f'Chain is currently {chain["status"]}, but should be unavailable to perform drop')
		self.drop_subject(chain['subject_a'], update_chain=False)
		self.drop_subject(chain['subject_b'], update_chain=False)
		if do_not_reopen:
			update_status = 'closed'
		else:
			update_status = 'available'
		self.db.chains.update_one({'chain_id': chain['chain_id']}, {'$set':{'status': update_status, 'subject_a': None, 'subject_b': None}})

	def review_subject(self, sub_id=None):
		if sub_id is None:
			return None
		subject = self.db.subjects.find_one({'subject_id': sub_id})
		if subject is None:
			raise ValueError('Subject not found')
		if subject['status'] not in ['approval_needed', 'reviewed']:
			raise ValueError('Subject not yet awaiting approval')
		minutes = (subject['modified_time'] - subject['creation_time']) // 60
		seconds = (subject['modified_time'] - subject['creation_time']) % 60
		lexicon = {}
		response_times = []
		attention_checks_passed = 0
		button_distributon = np.zeros(len(subject['input_lexicon']))
		object_distributon = np.zeros(len(subject['input_lexicon']))
		item_to_index_map = sorted(list(subject['input_lexicon'].keys()))
		for trial in subject['responses']:
			match trial['test_type']:
				case 'mini_test':
					if trial['catch_trial']:
						attention_checks_passed += trial['object_clicked']
					response_times.append(trial['response_time'])
				case 'test_production' | 'comm_production':
					trial['item'] = f'{trial["shape"]}_{trial["color"]}'
					lexicon[trial['item']] = trial['input_label']
					response_times.append(trial['response_time'])
				case 'test_comprehension' | 'comm_comprehension':
					button_distributon[trial['selected_button']] += 1
					object_distributon[item_to_index_map.index(trial['selected_item'])] += 1
					response_times.append(trial['response_time'])
		lexicon = {item: lexicon[item] for item in sorted(lexicon)}
		for item, word in lexicon.items():
			taught_word = subject['input_lexicon'][item]
			correct = '✅' if word == taught_word else '❌'
			trained = '➡️ ' if item in subject['training_items'] else '  '
			print(item, taught_word.ljust(9, ' '), trained, word.ljust(9, ' '), correct)
		self.db.subjects.update_one({'subject_id': sub_id}, {'$set':{'status': 'reviewed', 'lexicon': lexicon}})
		print('Time taken:', f'{minutes}:{str(seconds).zfill(2)}')
		print('Mean response time', round(np.mean(response_times) / 1000, 2))
		print('Attention checks:', attention_checks_passed)
		print('Button entropy:', round(entropy(button_distributon), 2))
		print('Object entropy:', round(entropy(object_distributon), 2))
		print('Total bonus:', subject['total_bonus'])
		print('Comments:', subject['comments'])

	def approve_subject(self, sub_id=None):
		if sub_id is None:
			return None
		subject = self.db.subjects.find_one({'subject_id': sub_id})
		if subject is None:
			raise ValueError('Subject not found')
		if subject['status'] != 'reviewed':
			raise ValueError('Subject not yet reviewed or already approved')
		self.db.subjects.update_one({'subject_id': sub_id}, {'$set':{'status': 'approved'}})
		self._log_approval(subject['subject_id'], subject['total_bonus'])
		return subject

	def manual_approve_subject(self, sub_id=None):
		if sub_id is None:
			raise ValueError('Subject ID must be specified')
		subject = self.db.subjects.find_one({'subject_id': sub_id})
		if subject is None:
			raise ValueError('Subject not found')
		self._log_approval(sub_id, subject['total_bonus'])

	def reject_subject(self, sub_id=None):
		if sub_id is None:
			return None
		subject = self.db.subjects.find_one({'subject_id': sub_id})
		if subject is None:
			raise ValueError('Subject not found')
		if subject['status'] != 'reviewed':
			raise ValueError('Subject not yet reviewed or already approved')
		self.db.subjects.update_one({'subject_id': sub_id}, {'$set':{'status': 'rejected'}})
		self._log_approval(subject['subject_id'], subject['total_bonus'])
		return subject

	def drop_subject(self, sub_id=None, update_chain=True):
		'''
		Set the subject's status to dropout, remove them from their assigned
		chain, and set the chain's status to available so that the slot can be
		refilled.
		'''
		if sub_id is None:
			return None
		subject = self.db.subjects.find_one({'subject_id': sub_id})
		if subject is None:
			raise ValueError('Subject not found')
		self.db.subjects.update_one({'subject_id': sub_id}, {'$set':{'status': 'dropout'}})
		if update_chain:
			chain = self.db.chains.find_one({'chain_id': subject['chain_id']})
			if chain['subject_a'] == subject['subject_id']:
				self.db.chains.update_one({'chain_id': subject['chain_id']}, {'$set':{'status': 'available', 'subject_a': None}})
			elif chain['subject_b'] == subject['subject_id']:
				self.db.chains.update_one({'chain_id': subject['chain_id']}, {'$set':{'status': 'available', 'subject_b': None}})


def entropy(distribution):
	distribution /= distribution.sum()
	return -sum([p * np.log(p) for p in distribution if p > 0])

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


if __name__ == '__main__':

	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('exp_id', action='store', help='Experiment ID')
	parser.add_argument('action', action='store', help='Action to take')
	parser.add_argument('target', action='store', nargs='?', default=None, help='Chain or Subject ID (as appropriate for the action)')
	args = parser.parse_args()

	e = MissionControl(args.exp_id)
	getattr(e, args.action)(args.target)
