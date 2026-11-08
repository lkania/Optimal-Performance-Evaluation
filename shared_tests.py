# Shared test statistics and testing procedures used by Figures 1 and 2(a).
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from shared_utilities import (
	batch_test_statistics,
	expected_fingerprint_exact,
	upper_quantile,
	wasserstein_1d,
)


def plugin_test_statistic(
		counts,
		n_observations,
		null_ps,
		null_ws,
		t,
):
	"""
	Compute the Wasserstein plug-in statistic.
	
	Notes
	-----
	Equation (10) uses W1(pi_hat, pi0), where
	pi_hat = n^{-1} sum_i delta_{X_i/t}. The fingerprint counts provide the weights
	of pi_hat on the grid {0/t, ..., t/t}.
	"""
	grid_ps = (
			jnp.arange(
				start=0,
				stop=t + 1,
				step=1,
				dtype=null_ps.dtype,
			)
			/ t
	)

	grid_ws = counts.astype(null_ps.dtype) / n_observations

	test_statistic = wasserstein_1d(
		u_values=grid_ps,
		u_weights=grid_ws,
		v_values=null_ps,
		v_weights=null_ws,
	)

	return test_statistic


def plugin_test(
		key,
		null_counts,
		alt_counts,
		null_ps,
		null_ws,
		n_observations,
		t,
		alpha,
):
	"""
	Calibrate and apply the Wasserstein plug-in test in Equation (10).

	Notes
	-----
	The critical value is estimated from simulations under the null hypothesis.
	The test then rejects alternative datasets whose plug-in statistic exceeds
	that critical value.
	"""
	calibration_statistics = batch_test_statistics(
		counts=null_counts,
		test_statistic_fn=plugin_test_statistic,
		n_observations=n_observations,
		null_ps=null_ps,
		null_ws=null_ws,
		t=t,
	)

	critical_value = upper_quantile(
		values=calibration_statistics,
		alpha=alpha,
	)

	test_statistics = batch_test_statistics(
		counts=alt_counts,
		test_statistic_fn=plugin_test_statistic,
		n_observations=n_observations,
		null_ps=null_ps,
		null_ws=null_ws,
		t=t,
	)

	test_results = test_statistics >= critical_value

	return test_results.astype(jnp.int32), critical_value


def discretized_plugin_test_statistic(
		counts,
		n_observations,
		efingerprint,
):
	"""Compute W1(pi_hat, nu_t(pi0)), Equation (15)."""
	efingerprint = efingerprint.reshape(-1)
	empirical_fingerprint = (
			counts.reshape(-1).astype(efingerprint.dtype)
			/ n_observations
	)

	# The fingerprint contains every bin j = 0, ..., t.
	t = efingerprint.shape[0] - 1

	cdf_difference = jnp.cumsum(
		empirical_fingerprint - efingerprint
	)

	return jnp.sum(jnp.abs(cdf_difference[:-1])) / t


def discretized_plugin_test(
		key,
		null_counts,
		alt_counts,
		null_ps,
		null_ws,
		n_observations,
		t,
		alpha,
):
	"""
	Calibrate and apply the discretized plugin test in Equation (15).

	Notes
	-----
	The critical value is estimated from simulations under the null hypothesis.
	"""
	efingerprint = expected_fingerprint_exact(
		ps=null_ps,
		ws=null_ws,
		t=t,
	)

	calibration_statistics = batch_test_statistics(
		counts=null_counts,
		test_statistic_fn=discretized_plugin_test_statistic,
		n_observations=n_observations,
		efingerprint=efingerprint,
	)

	critical_value = upper_quantile(
		values=calibration_statistics,
		alpha=alpha,
	)

	test_statistics = batch_test_statistics(
		counts=alt_counts,
		test_statistic_fn=discretized_plugin_test_statistic,
		n_observations=n_observations,
		efingerprint=efingerprint,
	)

	test_results = test_statistics >= critical_value

	return test_results.astype(jnp.int32), critical_value


def chi_squared_test_statistic(
		counts,
		n_observations,
		efingerprint,
		t,
):
	"""
	Compute the classical Pearson chi-squared statistic for fingerprint counts.
	
	Notes
	-----
	The observed cells are n_j and the expected cells are n b_{j,t}(pi0), where
	the expected fingerprint b_{j,t}(pi0) is defined in Equation (7).
	"""
	counts = counts.reshape(-1)
	efingerprint = efingerprint.reshape(-1)

	expected_counts = n_observations * efingerprint
	differences = counts.astype(efingerprint.dtype) - expected_counts

	raw_cell_statistics = differences ** 2 / jnp.where(
		expected_counts > 0.0,
		expected_counts,
		1.0,
	)

	cell_statistics = jnp.where(
		expected_counts > 0.0,
		raw_cell_statistics,
		jnp.where(
			counts > 0,
			jnp.inf,
			0.0,
		),
	)

	test_statistic = jnp.sum(cell_statistics)

	return test_statistic


def chi_squared_test(
		key,
		null_counts,
		alt_counts,
		null_ps,
		null_ws,
		n_observations,
		t,
		alpha,
):
	"""
	Calibrate and apply the classical Pearson chi-squared baseline.

	Notes
	-----
	The expected fingerprint is evaluated exactly from Equation (7). The critical
	value is estimated from simulations under the null hypothesis.
	"""
	efingerprint = expected_fingerprint_exact(
		ps=null_ps,
		ws=null_ws,
		t=t,
	)

	calibration_statistics = batch_test_statistics(
		counts=null_counts,
		test_statistic_fn=chi_squared_test_statistic,
		n_observations=n_observations,
		efingerprint=efingerprint,
		t=t,
	)

	critical_value = upper_quantile(
		values=calibration_statistics,
		alpha=alpha,
	)

	test_statistics = batch_test_statistics(
		counts=alt_counts,
		test_statistic_fn=chi_squared_test_statistic,
		n_observations=n_observations,
		efingerprint=efingerprint,
		t=t,
	)

	test_results = test_statistics >= critical_value

	return test_results.astype(jnp.int32), critical_value


def debiased_chi_squared_test_statistic(
		counts,
		n_observations,
		efingerprint,
		t,
):
	"""
	Compute the debiased Pearson statistic in Equation (16).
	
	Notes
	-----
	The random centering subtracts
	p_hat_j(1-p_hat_j)/(n-1), matching the second normalized Kravchuk polynomial
	displayed below Equation (16). The denominator uses
	w_j = max{1/(t+1), b_j(1-b_j)}. Averaging instead of summing changes the
	statistic only by the fixed factor 1/(t+1).
	"""
	counts = counts.reshape(-1)
	efingerprint = efingerprint.reshape(-1)

	empirical_fingerprint = counts.astype(efingerprint.dtype) / n_observations

	correction = (
			empirical_fingerprint
			* (1.0 - empirical_fingerprint)
			/ (n_observations - 1)
	)

	centered_square = (
			(empirical_fingerprint - efingerprint) ** 2
			- correction
	)

	denominator = jnp.maximum(
		efingerprint * (1.0 - efingerprint),
		1.0 / (t + 1),
	)

	test_statistic = jnp.mean(
		centered_square / denominator
	)

	return test_statistic


def debiased_chi_squared_test(
		key,
		null_counts,
		alt_counts,
		null_ps,
		null_ws,
		n_observations,
		t,
		alpha,
):
	"""
	Calibrate and apply the debiased Pearson test in Equation (16).

	Notes
	-----
	The expected fingerprint is evaluated exactly from Equation (7). The critical
	value is estimated from simulations under the null hypothesis.
	"""
	efingerprint = expected_fingerprint_exact(
		ps=null_ps,
		ws=null_ws,
		t=t,
	)

	calibration_statistics = batch_test_statistics(
		counts=null_counts,
		test_statistic_fn=debiased_chi_squared_test_statistic,
		n_observations=n_observations,
		efingerprint=efingerprint,
		t=t,
	)

	critical_value = upper_quantile(
		values=calibration_statistics,
		alpha=alpha,
	)

	test_statistics = batch_test_statistics(
		counts=alt_counts,
		test_statistic_fn=debiased_chi_squared_test_statistic,
		n_observations=n_observations,
		efingerprint=efingerprint,
		t=t,
	)

	test_results = test_statistics >= critical_value

	return test_results.astype(jnp.int32), critical_value


def global_minimax_test(
		key,
		null_counts,
		alt_counts,
		null_ps,
		null_ws,
		n_observations,
		t,
		alpha,
):
	"""
	Apply the global minimax test in Equation (17).
	
	Notes
	-----
	The fingerprint l1 test in Equation (15) and the debiased Pearson test in
	Equation (16) are each run at level alpha/2. The global test rejects if either
	component rejects.
	"""
	key_l1, key_debiased = jax.random.split(key, 2)

	l1_reject, l1_critical_value = discretized_plugin_test(
		key=key_l1,
		null_counts=null_counts,
		alt_counts=alt_counts,
		null_ps=null_ps,
		null_ws=null_ws,
		n_observations=n_observations,
		t=t,
		alpha=alpha / 2,
	)

	debiased_reject, debiased_critical_value = debiased_chi_squared_test(
		key=key_debiased,
		null_counts=null_counts,
		alt_counts=alt_counts,
		null_ps=null_ps,
		null_ws=null_ws,
		n_observations=n_observations,
		t=t,
		alpha=alpha / 2,
	)

	test_results = jnp.logical_or(
		l1_reject.astype(bool),
		debiased_reject.astype(bool),
	)

	critical_value = jnp.array(
		[
			l1_critical_value,
			debiased_critical_value,
		]
	)

	return test_results.astype(jnp.int32), critical_value