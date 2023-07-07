import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
from utils import json_load


def is_regular(lexicon, training_items):
	suffixes = {'0': set(), '1': set(), '2': set(), '3': set()}
	for item, word in lexicon.items():
		if item not in training_items:
			continue
		color = item.split('_')[1]
		suffixes[color].add(word[4:])
	if sum([len(suffixes[s]) == 1 for s in suffixes]) == 4:
		print(suffixes)
		return True
	return False

def find_subjects_with_regular_input():
	for i in range(1, 398):
		subject_id = str(i).zfill(3)
		if subject_id in gen_ones:
			continue
		data = json_load(f'../data/exp2/subject_{subject_id}.json')
		if data['generation'] > 3:
			continue
		if data['status'] != 'approved':
			continue
		if is_regular(data['input_lexicon'], data['training_items']):
			print(data['subject_id'], data['chain_id'], data['generation'])

def get_correct(subject_id):
	data = json_load(f'../data/exp2/subject_{subject_id}.json')
	correct = [
		int(trial['expected_label'] == trial['input_label'])
		for trial in data['responses'] if trial['test_type'] == 'mini_test'
	]
	trial = list(range(len(correct)))
	return trial, correct

def fit_model(trial, correct):
	coords = {
		'trial': trial,
	}
	with pm.Model(coords=coords) as model:
		α = pm.Normal('α', 0, 10)
		β = pm.Normal('β', 0, 10)
		p = pm.Deterministic('p', pm.math.invlogit(α + β * trial), dims='trial')
		y = pm.Bernoulli('y', p, observed=correct, dims='trial')
		trace = pm.sample(1000, tune=2000)
	return trace

def make_plot(axis, trace):
	post_mean = trace.posterior.p.mean(('chain', 'draw'))
	trial = range(1, len(post_mean) + 1)

	if post_mean[-1] > 0.75:
		color = 'MediumSeaGreen'
	else:
		color = 'Crimson'

	az.plot_hdi(
	    trial,
	    trace.posterior.p,
	    hdi_prob=0.95,
	    fill_kwargs={'alpha': 0.25, 'linewidth': 0},
	    ax=axis,
	    color=color,
	)

	axis.plot(trial, post_mean, label='posterior mean', color=color)
	axis.set_xticks([1, 12, 24, 36])
	axis.set_xticklabels([1, 12, 24, 36])
	axis.set_ylim(0, 1)
	axis.set_xlim(1, 36)

def plot_all_subjects(subject_ids):
	fig, axes = az.utils.plt.subplots(5, 10, figsize=(16, 9))
	for subject_id, axis in zip(subject_ids, np.ravel(axes)):
		trace = az.from_netcdf(f'../plots/exp2/training_traces/{subject_id}.netcdf')
		make_plot(axis, trace)
		axis.set_title(subject_id)
	fig.tight_layout()
	fig.savefig('../plots/exp2_learning_curves.pdf')

def run_all_subjects(subject_ids):
	for subject_id in subject_ids:
		trial, correct = get_correct(subject_id)
		trace = fit_model(trial, correct)
		trace.to_netcdf(f'../plots/exp2/training_traces/{subject_id}.netcdf')


if __name__ == '__main__':

	subjects_with_regular_input = ['008', '007', '010', '011', '009', '012', '013', '015', '026', '014', '017', '016', '018', '020', '033', '032', '021', '023', '030', '027', '002', '003', '004', '005', '006', '001', '024', '022', '028', '031',      '025', '035', '042', '045', '047', '048', '051', '052', '053', '054', '056', '058', '059', '062', '072', '077', '086', '087', '063', '076']

	# subjects_with_regular_input = subjects_with_regular_input[0:13]
	# subjects_with_regular_input = subjects_with_regular_input[13:26]
	# subjects_with_regular_input = subjects_with_regular_input[26:39]
	# subjects_with_regular_input = subjects_with_regular_input[39:50]

	# run_all_subjects(subjects_with_regular_input)
	plot_all_subjects(subjects_with_regular_input)
