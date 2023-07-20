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

def get_correct(exp_id, subject_id):
	data = json_load(f'../data/{exp_id}/subject_{subject_id}.json')
	correct = [
		int(trial['expected_label'] == trial['input_label'])
		for trial in data['responses'] if trial['test_type'] == 'mini_test'
	]
	prod_correct = [
		int(trial['expected_label'] == trial['input_label'])
		for trial in data['responses'] if trial['test_type'] == 'test_production'
	]
	comp_correct = [
		int(trial['selected_item'] in trial['items'])
		for trial in data['responses'] if trial['test_type'] == 'test_comprehension'
	]
	trial = list(range(len(correct)))
	return trial, correct, (sum(prod_correct), sum(comp_correct))

def fit_multilevel_model(trial, correct, subject_ids):
	coords = {
		'trial': list(range(1, 37)),
		'subject': subject_ids,
	}
	with pm.Model(coords=coords) as model:

		μ_α = pm.Normal('μ_α', 0, 10)
		σ_α = pm.Exponential('σ_α', 1)

		μ_β = pm.Normal('μ_β', 0, 10)
		σ_β = pm.Exponential('σ_β', 1)

		α = pm.Normal('α', μ_α, σ_α, dims='subject').T
		β = pm.Normal('β', μ_β, σ_β, dims='subject').T
		
		θ = pm.Deterministic('θ', pm.math.invlogit(α + β * trial), dims=('trial', 'subject'))

		c = pm.Bernoulli('c', θ, observed=correct, dims=('trial', 'subject'))

		pm.Deterministic('θ_mean', pm.math.invlogit(μ_α + μ_β * trial[:, 0]))
		
		trace = pm.sample(1000, tune=2000)

	return trace

def make_plot(axis, trace, sub_i=None):
	if sub_i is None:
		post = trace.posterior.θ_
	else:
		post = trace.posterior.θ[:, :, :, sub_i]
	post_mean = post.mean(('chain', 'draw'))
	trial = range(1, len(post_mean) + 1)

	if post_mean[-1] > 0.75:
		color = 'MediumSeaGreen'
	else:
		color = 'Crimson'

	az.plot_hdi(
	    trial,
	    post,
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

def run_multilevel(exp_id, subject_ids):
	trial_matrix = []
	correct_matrix = []
	for subject_id in subject_ids:
		trial, correct, _ = get_correct(exp_id, subject_id)
		trial_matrix.append(trial)
		correct_matrix.append(correct)
	trace = fit_multilevel_model(np.column_stack(trial_matrix), np.column_stack(correct_matrix), subject_ids)
	trace.to_netcdf('../plots/pilot8/training_trace.netcdf')

def plot_multilevel(exp_id, subject_ids):
	trace = az.from_netcdf('../plots/pilot8/training_trace.netcdf')
	fig, axes = az.utils.plt.subplots(4, 5, figsize=(16, 9))
	for sub_i, axis in zip(range(20), np.ravel(axes)):
		_, _, test_correct = get_correct(exp_id, subject_ids[sub_i])
		make_plot(axis, trace, sub_i)
		axis.set_title(subject_ids[sub_i] + f', test score = {test_correct}')
	fig.tight_layout()
	fig.savefig('../plots/pilot8_learning_curves.pdf')


if __name__ == '__main__':

	# exp2_subjects_with_regular_input = ['008', '007', '010', '011', '009', '012', '013', '015', '026', '014', '017', '016', '018', '020', '033', '032', '021', '023', '030', '027', '002', '003', '004', '005', '006', '001', '024', '022', '028', '031',      '025', '035', '042', '045', '047', '048', '051', '052', '053', '054', '056', '058', '059', '062', '072', '077', '086', '087', '063', '076']

	subject_ids = ['001', '002', '004', '003', '005', '006', '009', '010', '013', '014', '012', '015', '008', '016', '017', '011', '018', '007', '019', '020']

	# run_multilevel('pilot8', subject_ids)
	plot_multilevel('pilot8', subject_ids)
