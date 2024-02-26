from pathlib import Path
import numpy as np
import pandas as pd
import bambi as bmb
import xarray as xr


ROOT = Path(__file__).parent.parent.resolve()
DATA = ROOT / 'data'


def invlogit(a):
	return 1 / (1 + np.exp(-a))

def append_generational_model_to_trace(trace, apply_invlogit_transform=False):
	x = xr.DataArray(np.arange(1, 10))
	trace.add_groups({'constant_data': {'x1': x}})
	trace.constant_data['x'] = x
	theta = trace.posterior['α'] + trace.posterior['β'] * np.log(trace.constant_data['x'] + 1)
	if apply_invlogit_transform:
		trace.posterior['y_model'] = (('chain', 'draw', 'gen'), invlogit(theta).data)
	else:
		trace.posterior['y_model'] = (('chain', 'draw', 'gen'), theta.data)

def append_epochal_model_to_trace(trace, apply_invlogit_transform=False):
	x = xr.DataArray(np.arange(1, 10))
	trace.add_groups({'constant_data': {'x1': x}})
	trace.constant_data['x'] = x
	theta = []
	for epoch in range(1, 4):
		for generation_in_epoch in range(1, 4):
			theta.append(trace.posterior['α'] + \
				trace.posterior['β'] * epoch + \
				trace.posterior['γ'] * generation_in_epoch
			)
	theta = np.transpose(np.array(theta), (1, 2, 0))
	if apply_invlogit_transform:
		trace.posterior['y_model'] = (('chain', 'draw', 'gen'), invlogit(theta))
	else:
		trace.posterior['y_model'] = (('chain', 'draw', 'gen'), theta)


if __name__ == '__main__':


	df = pd.read_csv(DATA / 'exp.csv')


	# Experiment 1

	for condition in ['dif_lrn', 'dif_com', 'sil_com']:

		# Probability informative

		model = bmb.Model(
			formula='informative ~ np.log(generation + 1) + (1 + np.log(generation + 1) | chain)',
			data=df[ (df['condition'] == condition) & (df['generation'] >= 0) ],
			family='bernoulli',
		)
		model.set_alias({'Intercept': 'α', 'np.log(generation + 1)': 'β'})
		trace = model.fit(draws=12000, tune=2000, chains=6, cores=6, target_accept=0.95)
		append_generational_model_to_trace(trace, apply_invlogit_transform=True)
		trace.to_netcdf(DATA / 'models' / f'{condition}_pinf.netcdf')

		# Communicative cost

		model = bmb.Model(
			formula='cost ~ np.log(generation + 1) + (1 + np.log(generation + 1) | chain)',
			data=df[ (df['condition'] == condition) & (df['generation'] >= 0) ],
			family='gaussian',
		)
		model.set_alias({'Intercept': 'α', 'np.log(generation + 1)': 'β'})
		trace = model.fit(draws=12000, tune=2000, chains=6, cores=6, target_accept=0.95)
		append_generational_model_to_trace(trace, apply_invlogit_transform=False)
		trace.to_netcdf(DATA / 'models' / f'{condition}_cost.netcdf')

		# Transmission error

		model = bmb.Model(
			formula='error ~ np.log(generation + 1) + (1 + np.log(generation + 1) | chain)',
			data=df[ (df['condition'] == condition) & (df['generation'] >= 1) ],
			family='gaussian',
		)
		model.set_alias({'Intercept': 'α', 'np.log(generation + 1)': 'β'})
		trace = model.fit(draws=12000, tune=2000, chains=6, cores=6, target_accept=0.95)
		append_generational_model_to_trace(trace, apply_invlogit_transform=False)
		trace.to_netcdf(DATA / 'models' / f'{condition}_error.netcdf')


	# Experiment 2

	for condition in ['con_lrn', 'con_com']:

		# Probability informative

		model = bmb.Model(
			formula='informative ~ epoch + (1 + epoch | chain) + generation_in_epoch + (1 + generation_in_epoch | chain)',
			data=df[ (df['condition'] == condition) & (df['generation'] >= 1) ],
			family='bernoulli',
		)
		model.set_alias({'Intercept': 'α', 'epoch': 'β', 'generation_in_epoch': 'γ'})
		trace = model.fit(draws=12000, tune=2000, chains=6, cores=6, target_accept=0.95)
		append_epochal_model_to_trace(trace, apply_invlogit_transform=True)
		trace.to_netcdf(DATA / 'models' / f'{condition}_pinf.netcdf')

		# Communicative cost

		model = bmb.Model(
			formula='cost ~ epoch + (1 + epoch | chain) + generation_in_epoch + (1 + generation_in_epoch | chain)',
			data=df[ (df['condition'] == condition) & (df['generation'] >= 1) ],
			family='gaussian',
		)
		model.set_alias({'Intercept': 'α', 'epoch': 'β', 'generation_in_epoch': 'γ'})
		trace = model.fit(draws=12000, tune=2000, chains=6, cores=6, target_accept=0.95)
		append_epochal_model_to_trace(trace, apply_invlogit_transform=False)
		trace.to_netcdf(DATA / 'models' / f'{condition}_cost.netcdf')

		# Transmission error

		model = bmb.Model(
			formula='error ~ epoch + (1 + epoch | chain) + generation_in_epoch + (1 + generation_in_epoch | chain)',
			data=df[ (df['condition'] == condition) & (df['generation'] >= 1) ],
			family='gaussian',
		)
		model.set_alias({'Intercept': 'α', 'epoch': 'β', 'generation_in_epoch': 'γ'})
		trace = model.fit(draws=12000, tune=2000, chains=6, cores=6, target_accept=0.95)
		append_epochal_model_to_trace(trace, apply_invlogit_transform=False)
		trace.to_netcdf(DATA / 'models' / f'{condition}_error.netcdf')
