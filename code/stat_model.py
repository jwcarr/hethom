import pymc as pm
import arviz as az
import numpy as np

def generate_systems(n_systems, mu, sigma):
	return np.random.normal(mu, sigma, n_systems)

def paired_t_test(condition1, condition2):
	condition1 = np.array(condition1)
	condition2 = np.array(condition2)
	differences = condition1 - condition2
	with pm.Model() as model:
		μ = pm.Normal('μ', mu=500, sigma=100)
		σ = pm.Exponential('σ', lam=0.01)
		x = pm.Normal('x', mu=μ, sigma=σ, observed=differences)
		trace = pm.sample(5000, tune=1000, chains=4, cores=4, return_inferencedata=True)
	return trace

def unpaired_t_test(condition1, condition2):
	condition1 = np.array(condition1)
	condition2 = np.array(condition2)
	with pm.Model() as model:
		μ1 = pm.Normal('μ1', mu=350, sigma=100)
		σ1 = pm.Exponential('σ1', lam=0.01)
		x1 = pm.Normal('x1', mu=μ1, sigma=σ1, observed=condition1)
		μ2 = pm.Normal('μ2', mu=350, sigma=100)
		σ2 = pm.Exponential('σ2', lam=0.01)
		x2 = pm.Normal('x2', mu=μ2, sigma=σ2, observed=condition2)
		Δ = pm.Deterministic('Δ', μ1 - μ2)
		trace = pm.sample(5000, tune=1000, chains=4, cores=4, return_inferencedata=True)
	return trace


if __name__ == '__main__':

	condition1 = generate_systems(10, 250, 50)
	condition2 = generate_systems(10, 350, 50)

	trace = unpaired_t_test(condition1, condition2)

	az.plot_posterior(trace, hdi_prob=0.99, var_names=['μ1', 'μ2', 'Δ'])
	az.utils.plt.show()
