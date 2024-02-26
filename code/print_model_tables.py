import arviz as az
import pandas as pd

HDI_PROB = 0.95

def append_difference1(trace1, trace2):
	trace2.posterior['Δ(α)'] = trace2.posterior['α'] - trace1.posterior['α']
	trace2.posterior['Δ(β)'] = trace2.posterior['β'] - trace1.posterior['β']

def append_difference2(trace1, trace2):
	trace2.posterior['Δ(α)'] = trace2.posterior['α'] - trace1.posterior['α']
	trace2.posterior['Δ(β)'] = trace2.posterior['β'] - trace1.posterior['β']
	trace2.posterior['Δ(γ)'] = trace2.posterior['γ'] - trace1.posterior['β']

# Experiment 1

for measure in ['pinf', 'cost', 'error']:

	print(measure)

	trace1 = az.from_netcdf(f'../data/models/dif_lrn_{measure}.netcdf')
	trace2 = az.from_netcdf(f'../data/models/dif_com_{measure}.netcdf')

	append_difference1(trace1, trace2)

	table1 = az.summary(trace1, var_names=['α', 'β'], hdi_prob=HDI_PROB, round_to=2).drop(['mcse_mean', 'mcse_sd', 'ess_tail'], axis=1)
	table1.rename(index={'α': r'$\alpha_\mathrm{trans}$', 'β': r'$\beta_\mathrm{trans}$'}, inplace=True)

	table2 = az.summary(trace2, var_names=['α', 'β', 'Δ(α)', 'Δ(β)'], hdi_prob=HDI_PROB, round_to=2).drop(['mcse_mean', 'mcse_sd', 'ess_tail'], axis=1)
	table2.rename(index={
		'α': r'$\alpha_\mathrm{comm}$',
		'β': r'$\beta_\mathrm{comm}$',
		'Δ(α)': r'$\Delta(\alpha)$',
		'Δ(β)': r'$\Delta(\beta)$',
	}, inplace=True)

	table = pd.concat((table1, table2))

	table = table.astype({'ess_bulk':int})

	print(table.to_latex().replace('0000 ', ' '))
	print()


# Experiment 2

for measure in ['pinf', 'cost', 'error']:

	print(measure)

	trace1 = az.from_netcdf(f'../data/models/con_lrn_{measure}.netcdf')
	trace2 = az.from_netcdf(f'../data/models/con_com_{measure}.netcdf')

	append_difference2(trace1, trace2)

	table1 = az.summary(trace1, var_names=['α', 'β', 'γ'], hdi_prob=HDI_PROB, round_to=2).drop(['mcse_mean', 'mcse_sd', 'ess_tail'], axis=1)
	table1.rename(index={
		'α': r'$\alpha_\mathrm{trans}$',
		'β': r'$\beta_\mathrm{trans}$',
		'γ': r'$\gamma_\mathrm{trans}$',
	}, inplace=True)

	table2 = az.summary(trace2, var_names=['α', 'β', 'γ', 'Δ(α)', 'Δ(β)', 'Δ(γ)'], hdi_prob=HDI_PROB, round_to=2).drop(['mcse_mean', 'mcse_sd', 'ess_tail'], axis=1)
	table2.rename(index={
		'α': r'$\alpha_\mathrm{comm}$',
		'β': r'$\beta_\mathrm{comm}$',
		'γ': r'$\gamma_\mathrm{comm}$',
		'Δ(α)': r'$\Delta(\alpha)$',
		'Δ(β)': r'$\Delta(\beta)$',
		'Δ(γ)': r'$\Delta(\gamma)$',
	}, inplace=True)

	table = pd.concat((table1, table2))

	table = table.astype({'ess_bulk':int})

	print(table.to_latex().replace('0000 ', ' '))
	print()


# Experiment 3

for measure in ['pinf', 'cost', 'error']:

	print(measure)

	trace1 = az.from_netcdf(f'../data/models/dif_com_{measure}.netcdf')
	trace2 = az.from_netcdf(f'../data/models/sil_com_{measure}.netcdf')

	append_difference1(trace1, trace2)

	table1 = az.summary(trace1, var_names=['α', 'β'], hdi_prob=HDI_PROB, round_to=2).drop(['mcse_mean', 'mcse_sd', 'ess_tail'], axis=1)
	table1.rename(index={'α': r'$\alpha_\mathrm{exp1}$', 'β': r'$\beta_\mathrm{exp1}$'}, inplace=True)

	table2 = az.summary(trace2, var_names=['α', 'β', 'Δ(α)', 'Δ(β)'], hdi_prob=HDI_PROB, round_to=2).drop(['mcse_mean', 'mcse_sd', 'ess_tail'], axis=1)
	table2.rename(index={
		'α': r'$\alpha_\mathrm{exp3}$',
		'β': r'$\beta_\mathrm{exp3}$',
		'Δ(α)': r'$\Delta(\alpha)$',
		'Δ(β)': r'$\Delta(\beta)$',
	}, inplace=True)

	table = pd.concat((table1, table2))

	table = table.astype({'ess_bulk':int})

	print(table.to_latex().replace('0000 ', ' '))
	print()
