import pymc as pm
import arviz as az
import numpy as np

def generate_systems(n_systems, mu, sigma):
	return np.random.normal(mu, sigma, n_systems)

def unpaired_t_test(condition1, condition2, measure):
	condition1 = np.array(condition1)
	condition2 = np.array(condition2)

	with pm.Model(coords={'condition': [0, 1]}) as model:
		if measure == 'complexity':
			μ = pm.Uniform('μ', np.array([100, 100]), np.array([700, 700]), dims='condition')
			σ = pm.Exponential('σ', lam=np.array([0.01, 0.01]), dims='condition')
		else:
			μ = pm.Uniform('μ', np.array([0, 0]), np.array([2, 2]), dims='condition')
			σ = pm.Exponential('σ', lam=np.array([1, 1]), dims='condition')

		x1 = pm.TruncatedNormal('x1', mu=μ[0], sigma=σ[0], lower=0, upper=2, observed=condition1)
		x2 = pm.TruncatedNormal('x2', mu=μ[1], sigma=σ[1], lower=0, upper=2, observed=condition2)
		
		Δ = pm.Deterministic('Δ', μ[0] - μ[1])
		
		trace = pm.sample(5000, tune=1000, chains=4, cores=4, return_inferencedata=True)
	return trace


if __name__ == '__main__':

	# H1
	condition1 = generate_systems(10, 200, 30)
	condition2 = generate_systems(10, 600, 8)
	measure = 'complexity'

	# H2
	condition1 = np.array([0, 0, 0, 0, 0.1, 0.5, 0.4, 0.1, 0, 0.05])
	condition2 = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
	measure = 'cost'

	print(condition1.mean())
	print(condition2.mean())

	trace = unpaired_t_test(condition1, condition2, measure=measure)

	az.plot_posterior(trace, hdi_prob=0.99, var_names=['μ', 'σ', 'Δ'])
	az.utils.plt.show()
