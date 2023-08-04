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

Prior to running the experiment, create a subscription to the Prolific
submission.status.change event by running:

	create_subscription()

Make a note of the subscription_id and at the end of the experiment,
delete the subscription.

	delete_subscription('<subscription_id>')

'''

import re
import json
import random
import time
import requests
from itertools import product
from pathlib import Path
from pymongo import MongoClient
from bson.json_util import dumps


CONFIG_DIR = Path('config')
DATA_DIR = Path('../data')
DOMAIN = 'localhost'
PORT = 27017
PROLIFIC_CREDENTIALS_FILE = Path.home() / '.prolific_credentials.json'

DB = MongoClient(DOMAIN, PORT)

STATUS_EMOJI = {
	'available': '🟢',
	'unavailable': '🔴',
	'converged': '⚫️',
	'completed': '⚫️',
	'closed': '🟡',
	'approval_needed': '🔵',
}
EMPTY_SLOT = ' - ' * 8


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
				lexicon, spoken_forms = create_lexicon(task)
				DB[self.exp_id].chains.insert_one({
					'chain_id': f'{task["task_id"]}_{chain_i}',
					'task': task,
					'status': 'available',
					'current_gen': 0,
					'sound_epoch': 0,
					'subject_a': None,
					'subject_b': None,
					'subjects': [],
					'lexicon': lexicon,
					'spoken_forms': spoken_forms,
				})
		self.db = DB[self.exp_id]
		print('Launched task:', self.exp_id)

	def erase(self, _=None):
		if input(f'Are you sure you want to completely erase database "{self.exp_id}"? ') == 'yes':
			DB.drop_database(self.exp_id)
			self.db = None

	def status(self, _=None):
		n_available = 0
		for chain in self.db.chains.find():
			if chain['task']['communication']:
				print(chain['chain_id'], str(chain['current_gen']).rjust(2, ' '), STATUS_EMOJI[chain['status']], chain['status'].ljust(15), f'[{chain["subject_a"] or EMPTY_SLOT}]', f'[{chain["subject_b"] or EMPTY_SLOT}]')
			else:
				print(chain['chain_id'], str(chain['current_gen']).rjust(2, ' '), STATUS_EMOJI[chain['status']], chain['status'].ljust(15), f'[{chain["subject_a"] or EMPTY_SLOT}]')
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
			anon_subject_id = str(i).zfill(3)
			subject_id_map[ subject['subject_id'] ] = anon_subject_id
			subject['subject_id'] = anon_subject_id
			with open(exp_dir / f'subject_{anon_subject_id}.json', 'w') as file:
				file.write(dumps(subject, indent='\t'))
			subject_data[anon_subject_id] = subject
		for chain in self.db.chains.find({}):
			del chain['_id']
			chain['subjects'] = [
				(subject_id_map[subject_a], subject_id_map[subject_b]) for subject_a, subject_b in chain['subjects']
			]
			with open(exp_dir / f'chain_{chain["chain_id"]}.json', 'w') as file:
				file.write(dumps(chain, indent='\t'))

	def open(self, chain_id):
		if chain_id is None:
			raise ValueError('Chain ID must be specified')
		pattern = re.compile(chain_id)
		for chain in self.db.chains.find():
			if pattern.fullmatch(chain['chain_id']):
				if chain['status'] == 'closed':
					self.db.chains.update_one({'chain_id': chain['chain_id']}, {'$set': {'status': 'available'}})
					print(f'Opened {chain["chain_id"]}')
				else:
					print(f'Cannot open {chain["chain_id"]}: Chain not currently closed')

	def close(self, chain_id=None):
		if chain_id is None:
			raise ValueError('Chain ID must be specified')
		pattern = re.compile(chain_id)
		for chain in self.db.chains.find():
			if pattern.fullmatch(chain['chain_id']):
				if chain['subject_a'] is None and chain['subject_b'] is None and chain['status'] == 'available':
					self.db.chains.update_one({'chain_id': chain['chain_id']}, {'$set': {'status': 'closed'}})
					print(f'Closed {chain["chain_id"]}')
				else:
					print(f'Cannot close {chain["chain_id"]}: Chain occupied')

	def review(self):
		for chain in self.db.chains.find({'status': 'approval_needed'}):
			print(f'CHAIN: {chain["chain_id"]}')
			print()
			self.review_subject(chain['subject_a'])
			self.review_subject(chain['subject_b'])
			if input('Do you want to approve this generation? ') == 'yes':
				self.approve(chain['chain_id'])

	def approve(self, chain_id=None):
		if chain_id is None:
			raise ValueError('Chain ID must be specified')
		chain = self.db.chains.find_one({'chain_id': chain_id})
		if chain is None:
			raise ValueError('Chain not found')
		if chain['status'] != 'approval_needed':
			raise ValueError('Chain not awaiting approval')
		subject_a = self.approve_subject(chain['subject_a'])
		subject_b = self.approve_subject(chain['subject_b'])
		if subject_a_is_dominant(subject_a, subject_b):
			next_lexicon = subject_a['lexicon']
		else:
			next_lexicon = subject_b['lexicon']
		if input('Do you want to reopen this chain? ') == 'yes':
			update_status = 'available'
		else:
			update_status = 'closed'
		if chain['current_gen'] + 1 >= chain['task']['max_gens']:
			update_status = 'completed'
			print('🎉 CHAIN COMPLETED!')
		elif chain['task']['stop_on_convergence']:
			for item, word in chain['lexicon'].items():
				if next_lexicon[item] != word:
					break
			else: # for loop exits normally, all words match, chain has converged
				update_status = 'converged'
				print('🎉 CHAIN CONVERGED!')
		sound_epoch_inc = 0
		if chain['task']['sound_change_freq']:
			if (chain['current_gen'] > 0 and (chain['current_gen'] + 1) % chain['task']['sound_change_freq'] == 0) or (chain['task']['sound_change_freq'] == 1):
				sound_epoch_inc = 1
				print('🕓 NEW SOUND EPOCH')
		self.db.chains.update_one({'chain_id': chain['chain_id']}, {
			'$set': {'status': update_status, 'lexicon': next_lexicon, 'subject_a': None, 'subject_b': None},
			'$push': {'subjects': [chain['subject_a'], chain['subject_b']]},
			'$inc': {'current_gen': 1, 'sound_epoch': sound_epoch_inc},
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
		update_status = 'closed' if do_not_reopen else 'available'
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
		update_status = 'closed' if do_not_reopen else 'available'
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
		training_correct = []
		for trial in subject['responses']:
			match trial['test_type']:
				case 'mini_test':
					if trial['catch_trial']:
						attention_checks_passed += trial['object_clicked']
					response_times.append(trial['response_time'])
					training_correct.append(trial['expected_label'] == trial['input_label'])
				case 'test_production' | 'comm_production':
					trial['item'] = f'{trial["shape"]}_{trial["color"]}'
					lexicon[trial['item']] = trial['input_label']
					response_times.append(trial['response_time'])
				case 'test_comprehension' | 'comm_comprehension':
					response_times.append(trial['response_time'])
		lexicon = {item: lexicon[item] for item in sorted(lexicon)}
		for item, word in lexicon.items():
			taught_word = subject['input_lexicon'][item]
			correct = '✅' if word == taught_word else '❌'
			trained = '➡️ ' if item in subject['training_items'] else '  '
			print(item, taught_word.ljust(9, ' '), trained, word.ljust(9, ' '), correct, subject['spoken_forms'][item])
		self.db.subjects.update_one({'subject_id': sub_id}, {'$set':{'status': 'reviewed', 'lexicon': lexicon}})
		print('Subject ID:', subject['subject_id'])
		print('Time taken:', f'{minutes}:{str(seconds).zfill(2)}')
		print('Mean response time', round(sum(response_times) / len(response_times) / 1000, 2))
		print('Attention checks:', attention_checks_passed)
		print('Training score:', round(sum(training_correct[-12:]) / 12 * 100, 0))
		print('Total bonus:', subject['total_bonus'])
		print('Comments:', subject['comments'])
		print()

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

	def drop_subject(self, sub_id=None, update_chain=True, pay_subject=None, ignore_subject_not_found=False):
		'''
		Set the subject's status to dropout, remove them from their assigned
		chain, and set the chain's status to available so that the slot can be
		refilled.
		'''
		if sub_id is None:
			return None
		subject = self.db.subjects.find_one({'subject_id': sub_id})
		if subject is None:
			if ignore_subject_not_found:
				return
			raise ValueError('Subject not found')
		if pay_subject is None:
			pay_subject = input(f'Should subject {sub_id} get paid? ') == 'yes'
		if pay_subject:
			self._log_approval(sub_id, subject['total_bonus'])
			self.db.subjects.update_one({'subject_id': sub_id}, {'$set':{'status': 'jilted'}})
			print(f'Status of {sub_id} changed to jilted')
		else:
			self.db.subjects.update_one({'subject_id': sub_id}, {'$set':{'status': 'dropout'}})
			print(f'Status of {sub_id} changed to dropout')
		if update_chain and subject['chain_id']:
			chain = self.db.chains.find_one({'chain_id': subject['chain_id']})
			if chain['subject_a'] == subject['subject_id']:
				self.db.chains.update_one({'chain_id': subject['chain_id']}, {'$set':{'status': 'available', 'subject_a': None}})
				print(f'Subject A slot opened on chain {chain["chain_id"]}')
			elif chain['subject_b'] == subject['subject_id']:
				self.db.chains.update_one({'chain_id': subject['chain_id']}, {'$set':{'status': 'available', 'subject_b': None}})
				print(f'Subject B slot opened on chain {chain["chain_id"]}')


def choose(choices, choose_n):
	'''
	Choose N items from choices without replacement.
	'''
	assert choose_n <= len(choices)
	random.shuffle(choices)
	return choices[:choose_n]

def create_random_lexicon_with_variation(task):
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
	lexicon = {}
	spoken_forms = {}
	for i in range(task['n_shapes']):
		for j in range(task['n_colors']):
			item = f'{i}_{j}'
			lexicon[item] = task['stems'][i] + endings.pop()
			spoken_forms[item] = f'{i}.m4a'
	return lexicon, [spoken_forms]

def create_compositional_lexicon_with_sound_change(task):
	assert task['n_shapes'] == len(task['stems'])
	assert len(task['seed_suffix_spellings']) == 2
	assert all([task['n_colors'] == len(spellings) for spellings in task['seed_suffix_spellings']])

	cons = list(range(task['n_colors']))
	vwls = list(range(task['n_colors']))
	random.shuffle(cons)
	random.shuffle(vwls)

	lexicon = {}
	phonetic_lexicon = {}
	for i, stem in enumerate(task['stems']):
		for j, (c, v) in enumerate(zip(cons, vwls)):
			item = f'{i}_{j}'
			suffix = f'{task["seed_suffix_spellings"][0][c]}{task["seed_suffix_spellings"][1][v]}'
			lexicon[item] = stem + suffix
			phonetic_lexicon[(i, j)] = i, c, v
	phonetic_lexicons = [phonetic_lexicon]

	suffix_indices = list(range(task['n_colors']))
	random.shuffle(suffix_indices)
	for merger in [ suffix_indices[:2], suffix_indices[2:], suffix_indices ]:
		suffix1, suffix2 = set([sound[1:] for item, sound in phonetic_lexicons[-1].items() if item[1] in merger])
		if random.random() < 0.5:
			new_suffix = [suffix1[0], suffix2[1]]
		else:
			new_suffix = [suffix2[0], suffix1[1]]
		for m in merger:
			cons[m] = new_suffix[0]
			vwls[m] = new_suffix[1]
		phonetic_lexicon = {}
		for i, stem in enumerate(task['stems']):
			for j, (c, v) in enumerate(zip(cons, vwls)):
				phonetic_lexicon[(i, j)] = i, c, v
		phonetic_lexicons.append(phonetic_lexicon)

	phonetic_lexicons = [
		{'_'.join(map(str, item)): '_'.join(map(str, sound)) + '.m4a' for item, sound in phonetic_lexicon.items()}
		for phonetic_lexicon in phonetic_lexicons
	]
	return lexicon, phonetic_lexicons

def create_compdet_lexicon_with_sound_change(task):
	cons = list(range(task['n_colors']))
	vwls = list(range(task['n_colors']))
	random.shuffle(cons)
	random.shuffle(vwls)

	lexicon = {}
	phonetic_lexicon = {}
	for i, stem in enumerate(task['stems']):
		for j, (c, v) in enumerate(zip(cons, vwls)):
			item = f'{i}_{j}'
			suffix = f'{task["seed_suffix_spellings"][0][c]}{task["seed_suffix_spellings"][1][v]}'
			lexicon[item] = stem + suffix
			phonetic_lexicon[(i, j)] = i, c, v
	phonetic_lexicons = [phonetic_lexicon]

	suffix_indices = list(range(task['n_colors']))
	random.shuffle(suffix_indices)
	for merger in [ suffix_indices[:2], suffix_indices ]:
		suffix1, suffix2 = set([sound[1:] for item, sound in phonetic_lexicons[-1].items() if item[1] in merger])
		if random.random() < 0.5:
			new_suffix = [suffix1[0], suffix2[1]]
		else:
			new_suffix = [suffix2[0], suffix1[1]]
		for m in merger:
			cons[m] = new_suffix[0]
			vwls[m] = new_suffix[1]
		phonetic_lexicon = {}
		for i, stem in enumerate(task['stems']):
			for j, (c, v) in enumerate(zip(cons, vwls)):
				phonetic_lexicon[(i, j)] = i, c, v
		phonetic_lexicons.append(phonetic_lexicon)

	phonetic_lexicons = [
		{'_'.join(map(str, item)): '_'.join(map(str, sound)) + '.m4a' for item, sound in phonetic_lexicon.items()}
		for phonetic_lexicon in phonetic_lexicons
	]
	return lexicon, phonetic_lexicons

def create_holrand_lexicon_without_sound_change(task):
	suffix_spellings = []
	for suffix_spelling in product(*task['seed_suffix_spellings']):
		suffix_spellings.append(''.join(suffix_spelling))
	random.shuffle(suffix_spellings)
	lexicon = {}
	spoken_forms = {}
	for i in range(task['n_shapes']):
		for j in range(task['n_colors']):
			item = f'{i}_{j}'
			stem = task['stems'][i]
			suffix = suffix_spellings.pop()
			lexicon[item] = stem + suffix
			spoken_forms[item] = f'{i}.m4a'
	return lexicon, [spoken_forms]

def create_lexicon(task):
	if 'seed_suffix_system' in task:
		if task['seed_suffix_system'] == 'compositional_deterministic':
			return create_compdet_lexicon_with_sound_change(task)
		elif task['seed_suffix_system'] == 'holistic_random':
			return create_holrand_lexicon_without_sound_change(task)
		else:
			raise ValueError('Cannot construct lexicon: Invalid config specification')
	if 'lexicon' in task and 'spoken_forms' in task:
		return task['lexicon'], task['spoken_forms']
	if task['sound_change_freq']:
		return create_compositional_lexicon_with_sound_change(task)
	elif 'seed_suffix_variation' in task:
		return create_random_lexicon_with_variation(task)
	raise ValueError('Cannot construct lexicon: Invalid config specification')

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


try:
	with open(PROLIFIC_CREDENTIALS_FILE) as file:
		PROLIFIC_CREDENTIALS = json.load(file)
except FileNotFoundError:
	PROLIFIC_CREDENTIALS = None

def set_up_secret():
	response = requests.post(
		'https://api.prolific.co/api/v1/hooks/secrets/',
		headers={'Authorization': f'Token {PROLIFIC_CREDENTIALS["api_token"]}'},
		json={'workspace_id': PROLIFIC_CREDENTIALS['workspace_id']},
	).json()
	print(response)

def list_all_secrets():
	response = requests.get(
		'https://api.prolific.co/api/v1/hooks/secrets/',
		headers={'Authorization': f'Token {PROLIFIC_CREDENTIALS["api_token"]}'},
	).json()
	print(response)

def list_all_subscriptions():
	response = requests.get(
		'https://api.prolific.co/api/v1/hooks/subscriptions/?is_enabled=true',
		headers={'Authorization': f'Token {PROLIFIC_CREDENTIALS["api_token"]}'},
	).json()
	print(response)

def create_subscription():
	response = requests.post(
		'https://api.prolific.co/api/v1/hooks/subscriptions/',
		headers={'Authorization': f'Token {PROLIFIC_CREDENTIALS["api_token"]}'},
		json={'workspace_id': PROLIFIC_CREDENTIALS['workspace_id'], 'event_type': 'submission.status.change', 'target_url': 'https://joncarr.net:8080/prolific'},
	)
	subscription_id = response.json()['id']
	x_hook_secret = response.headers['X-Hook-Secret']
	print('Subscription ID:', subscription_id)
	print('X-Hook-Secret:', x_hook_secret)
	response = requests.post(
		f'https://api.prolific.co/api/v1/hooks/subscriptions/{subscription_id}/',
		headers={'Authorization': f'Token {PROLIFIC_CREDENTIALS["api_token"]}'},
		json={'secret': x_hook_secret, 'workspace_id': PROLIFIC_CREDENTIALS['workspace_id'], 'event_type': 'submission.status.change', 'target_url': 'https://joncarr.net:8080/prolific'},
	)
	print(response)

def delete_subscription(subscription_id):
	response = requests.delete(
		f'https://api.prolific.co/api/v1/hooks/subscriptions/{subscription_id}/',
		headers={'Authorization': f'Token {PROLIFIC_CREDENTIALS["api_token"]}'},
	)
	print(response)


if __name__ == '__main__':

	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('exp_id', action='store', help='Experiment ID')
	parser.add_argument('action', action='store', help='Action to take')
	parser.add_argument('target', action='store', nargs='?', default=None, help='Chain or Subject ID (as appropriate for the action)')
	args = parser.parse_args()

	e = MissionControl(args.exp_id)
	getattr(e, args.action)(args.target)
