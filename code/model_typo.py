from pathlib import Path
import numpy as np
import pymc as pm


ROOT = Path(__file__).parent.parent.resolve()
DATA = ROOT / 'data'


def construct_t_matrix(data_path, experiment):
	from utils import json_load
	import voi
	import matrix

	dataset = json_load(data_path)

	ref_systems = [
		matrix.reference_systems['holistic'],
		matrix.reference_systems['expressive'],
		matrix.reference_systems['redundant'],
		matrix.reference_systems['transparent'],
	]

	t = np.zeros((2, 9, 4), dtype=int)
	for i, condition in enumerate([f'{experiment}_lrn', f'{experiment}_com']):
		for j, generation in enumerate(range(1, 10)):
			for chain in dataset[condition]:
				subject_a = chain[generation][0]
				subject_system = matrix.make_matrix(subject_a['lexicon'], 3, 3)
				distances = [
					voi.variation_of_information(subject_system, ref_system)
					for ref_system in ref_systems
				]
				category = np.argmin(distances)
				t[i, j, category] += 1
	return t


if __name__ == '__main__':

	data_path = DATA / 'exp3.json'
	experiment = 'con'
	output_path = DATA / 'exp2_typo.netcdf'

	t = construct_t_matrix(data_path, experiment)

	coords = {
		'category': ['H', 'E', 'R', 'D'],
		'generation': list(range(1, 10)),
		'condition': ['lrn', 'com'],
	}

	with pm.Model(coords=coords) as model:

		# Prior
		θ = pm.Dirichlet('θ', a=np.ones(4), dims=('condition', 'generation', 'category'))

		# Likelihood
		pm.Multinomial('c', n=10, p=θ, observed=t, dims=('condition', 'generation', 'category'))

		# Deterministic parameter
		pm.Deterministic('diff_θ', θ[1, :, :] - θ[0, :, :], dims=('generation', 'category'))

		# Sampling
		trace = pm.sample(10000, tune=2000, chains=6, cores=6)
		trace.to_netcdf(output_path)
