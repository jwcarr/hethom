import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
from utils import json_load

def get_correct(subject_id):
	data = json_load(f'../data/exp2/subject_{subject_id}.json')
	correct = [
		int(trial['expected_label'][3:] == trial['input_label'][3:])
		for trial in data['responses'] if trial['test_type'] == 'mini_test'
	]
	trial = list(range(len(correct)))
	return trial, correct

def fit_model(trial, correct):
	coords = {
		'trial': trial,
	}
	with pm.Model(coords=coords) as model:
		α = pm.Normal('α', 0, 2)
		β = pm.Normal('β', 0, 2)
		p = pm.Deterministic('p', pm.math.invlogit(α + β * trial), dims='trial')
		y = pm.Bernoulli('y', p, observed=correct, dims='trial')
		trace = pm.sample(1000, tune=2000)
	return trace

def fit_all_gen1():
	dataset_json = json_load('../data/exp2.json')
	for condition, data in dataset_json.items():
		for chain in data:
			for generation in chain[1:2]:
				for subject in generation:
					if subject:
						subject_id = subject['subject_id']
						trial, correct = get_correct(subject_id)
						trace = fit_model(trial, correct)
						
						# fig, axis = az.utils.plt.subplots(1, 1)
						# make_plot(axis, trace)
						# az.utils.plt.show()
						# quit()
						
						trace.to_netcdf(f'../plots/exp2_traces/{subject_id}.netcdf')

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

def plot_all():
	fig, axes = az.utils.plt.subplots(5, 6, figsize=(16, 9))
	axes = np.ravel(axes)
	axis_i = 0
	dataset_json = json_load('../data/exp2.json')
	for condition, data in dataset_json.items():
		for chain in data:
			for generation in chain[1:2]:
				for subject in generation:
					if subject:
						subject_id = subject['subject_id']
						trace = az.from_netcdf(f'../plots/exp2_trace_stems/{subject_id}.netcdf')
						make_plot(axes[axis_i], trace)
						axes[axis_i].set_title(subject_id)
						axis_i += 1
	fig.tight_layout()
	fig.savefig('../plots/exp2_learning_curves_stems.pdf')



# fit_all_gen1()
plot_all()





