# Shared utility functions used by Figures 1, 2(a), and 2(b).
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy.stats import binom as jax_binom
from functools import partial


def wasserstein_1d(
		u_values,
		u_weights,
		v_values,
		v_weights,
):
	"""
	Compute the 1-Wasserstein distance between two finite distributions on the line.
	
	Notes
	-----
	Equation (8) defines W1. In one dimension, W1 equals the integral of the
	absolute difference between the two CDFs, which is evaluated exactly on the
	combined discrete support here.
	"""
	u_sorter = jnp.argsort(u_values)
	v_sorter = jnp.argsort(v_values)

	all_values = jnp.concatenate(
		(
			u_values,
			v_values,
		)
	)

	all_values = jnp.sort(all_values)

	deltas = jnp.diff(all_values)

	u_cdf_indices = jnp.searchsorted(
		u_values[u_sorter],
		all_values[:-1],
		side="right",
	)

	v_cdf_indices = jnp.searchsorted(
		v_values[v_sorter],
		all_values[:-1],
		side="right",
	)

	u_sorted_cumweights = jnp.concatenate(
		(
			jnp.array([0.0], dtype=u_weights.dtype),
			jnp.cumsum(u_weights[u_sorter]),
		)
	)

	v_sorted_cumweights = jnp.concatenate(
		(
			jnp.array([0.0], dtype=v_weights.dtype),
			jnp.cumsum(v_weights[v_sorter]),
		)
	)

	u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]
	v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

	return jnp.sum(jnp.abs(u_cdf - v_cdf) * deltas)


@partial(jax.jit, static_argnames=("n_datasets", "t"))
def sample_binomial_mixture_counts_batch(
		key,
		ps,
		ws,
		n_datasets,
		n_observations,
		t,
):
	# 	"""
	# 	Sample batches of datasets from the random-effects binomial model.
	#
	# 	Notes
	# 	-----
	# 	Equation (1) gives X_i | p_i ~ Bin(t, p_i) with p_i ~ pi. Each simulated
	# 	dataset is converted to fingerprint counts n_j = sum_i I(X_i = j), as in
	# 	Equation (12).
	# 	"""
	probabilities = expected_fingerprint_exact(
		ps=ps,
		ws=ws,
		t=t,
	)

	probabilities = jnp.broadcast_to(
		probabilities,
		(n_datasets, t + 1),
	)

	counts = jax.random.multinomial(
		key=key,
		n=n_observations,
		p=probabilities,
		dtype=jnp.float64,
	)

	return counts.astype(jnp.int64)


def expected_fingerprint_exact(
		ps,
		ws,
		t,
):
	"""
	Compute the expected fingerprint of a discrete mixing distribution exactly.
	
	Notes
	-----
	Equation (7) defines b_{j,t}(pi) = E_{p~pi}[B_{j,t}(p)]. Since every mixing
	distribution used by these simulations has finite support, the expectation can
	be evaluated directly as a weighted mixture of Bin(t, p) probability masses.
	"""
	xs = jnp.arange(
		start=0,
		stop=t + 1,
		step=1,
		dtype=ps.dtype,
	)

	binomial_probabilities = jax.vmap(
		lambda p: jax_binom.pmf(
			xs,
			t,
			p,
		)
	)(ps)

	efingerprint = jnp.sum(
		ws[:, None] * binomial_probabilities,
		axis=0,
	)

	return efingerprint / jnp.sum(efingerprint)


def upper_quantile(values, alpha):
	"""
	Compute an empirical upper-tail critical value.
	
	Notes
	-----
	For level alpha, the function returns an empirical (1-alpha)-quantile from
	simulations under the null hypothesis. This is the q_alpha notation used in
	Equations (10), (15), (16), (24), (25), and (33).
	"""
	values = jnp.sort(values)

	n_values = values.shape[0]

	index = jnp.ceil((1.0 - alpha) * n_values).astype(jnp.int32) - 1
	index = jnp.clip(
		index,
		0,
		n_values - 1,
	)

	return values[index]


def batch_test_statistics(
		counts,
		test_statistic_fn,
		**test_statistic_kwargs,
):
	"""
	Compute a test statistic for a batch of simulated datasets.

	Notes
	-----
	Each row of ``counts`` is the fingerprint of one simulated dataset.
	jax.vmap applies ``test_statistic_fn`` independently to every dataset.
	"""
	return jax.vmap(
		lambda count: test_statistic_fn(
			counts=count,
			**test_statistic_kwargs,
		)
	)(counts)


def homogeneity_null_distribution(p0):
	"""
	Construct the known-reference null distribution pi0 = delta_{p0}.
	
	Notes
	-----
	Equation (21) specifies H0: pi = delta_{p0}. The zero-weight duplicate support
	point keeps the JAX array shape equal to the two-point alternative.
	"""
	ps = jnp.array(
		[
			p0,
			p0,
		],
		dtype=jnp.float64,
	)

	ws = jnp.array(
		[
			1.0,
			0.0,
		],
		dtype=jnp.float64,
	)

	return ps, ws


def two_point_alternative(
		p0,
		epsilon,
):
	"""Construct pi = (delta_{p0-epsilon} + delta_{p0+epsilon}) / 2.

	The alternative has mean p0, variance epsilon**2, and
	W1(pi, delta_{p0}) = epsilon. It isolates the variance signal.
	"""
	if epsilon <= 0.0 or p0 - epsilon < 0.0 or p0 + epsilon > 1.0:
		raise ValueError("Symmetric alternative must be supported on [0, 1].")

	ps = jnp.array(
		[
			p0 - epsilon,
			p0 + epsilon,
		],
		dtype=jnp.float64,
	)

	ws = jnp.array(
		[
			0.5,
			0.5,
		],
		dtype=jnp.float64,
	)

	return ps, ws


def mean_shifted_alternative(p0, epsilon):
	"""Construct the original family (delta_p0 + delta_{p0+epsilon}) / 2.

	The alternative has mean p0 + epsilon/2, variance epsilon**2/4,
	and W1(pi, delta_{p0}) = epsilon/2.
	"""
	if epsilon <= 0.0 or p0 < 0.0 or p0 + epsilon > 1.0:
		raise ValueError("Shifted alternative must be supported on [0, 1].")

	return (
		jnp.array([p0, p0 + epsilon], dtype=jnp.float64),
		jnp.array([0.5, 0.5], dtype=jnp.float64),
	)