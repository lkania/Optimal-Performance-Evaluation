# Simulation for Figure 2(a).
from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pandas as pd

from shared_tests import (
	chi_squared_test,
	global_minimax_test,
	plugin_test,
)
from shared_utilities import (
	two_point_alternative,
	mean_shifted_alternative,
	batch_test_statistics,
	homogeneity_null_distribution,
	sample_binomial_mixture_counts_batch,
	upper_quantile,
)
from tqdm import tqdm

from plotnine import (
	ggplot,
	aes,
	element_text,
	geom_line,
	geom_point,
	facet_wrap,
	theme_bw,
	theme,
	labs,
	guides,
	guide_legend,
	scale_color_manual
)


def kravchuk1(
		x,
		p0,
		t,
):
	"""
	Evaluate the first normalized Kravchuk polynomial.

	Notes
	-----
	Equation (24) uses K_1(X_i, p0, t) = X_i/t - p0 to detect one-sided deviations
	of the mean from the known reference effect.
	"""
	return x / t - p0


def kravchuk2(
		x,
		p0,
		t,
):
	"""
	Evaluate the second normalized Kravchuk polynomial.

	Notes
	-----
	Equation (25) uses the second Kravchuk polynomial to estimate the squared
	deviation from p0 without binomial sampling bias.
	"""
	x_over_t = x / t

	conditional_variance_correction = (
			x_over_t
			* (1.0 - x_over_t)
			/ (t - 1.0)
	)

	return (
			(x_over_t - p0) ** 2
			- conditional_variance_correction
	)


def local_minimax_test_statistic(
		counts,
		n_observations,
		p0,
		t,
):
	"""
	Compute the two component statistics used by the local minimax test.

	Notes
	-----
	The first coordinate is the one-sided mean statistic in Equation (24). The
	second coordinate is the debiased variance statistic in Equation (25).
	"""
	xs = jnp.arange(
		start=0,
		stop=t + 1,
		step=1,
		dtype=counts.dtype,
	).astype(jnp.float64)

	props = counts.astype(jnp.float64) / n_observations

	mean_statistic = jnp.sum(
		kravchuk1(
			x=xs,
			p0=p0,
			t=t,
		)
		* props
	)

	variance_statistic = jnp.sum(
		kravchuk2(
			x=xs,
			p0=p0,
			t=t,
		)
		* props
	)

	return jnp.array(
		[
			mean_statistic,
			variance_statistic,
		]
	)


def local_minimax_test(
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
	Calibrate and apply the local minimax test in Equation (26).

	Notes
	-----
	The tests in Equations (24) and (25) are each calibrated at level alpha/2 using
	simulations under the null hypothesis. Equation (26) rejects if either
	component rejects.
	"""
	p0 = null_ps[0]

	calibration_statistics = batch_test_statistics(
		counts=null_counts,
		test_statistic_fn=local_minimax_test_statistic,
		n_observations=n_observations,
		p0=p0,
		t=t,
	)

	mean_critical_value = upper_quantile(
		values=calibration_statistics[:, 0],
		alpha=alpha / 2,
	)

	variance_critical_value = upper_quantile(
		values=calibration_statistics[:, 1],
		alpha=alpha / 2,
	)

	critical_value = jnp.array(
		[
			mean_critical_value,
			variance_critical_value,
		]
	)

	test_statistics = batch_test_statistics(
		counts=alt_counts,
		test_statistic_fn=local_minimax_test_statistic,
		n_observations=n_observations,
		p0=p0,
		t=t,
	)

	mean_reject = test_statistics[:, 0] >= mean_critical_value
	variance_reject = test_statistics[:, 1] >= variance_critical_value

	test_results = jnp.logical_or(
		mean_reject,
		variance_reject,
	)

	return test_results.astype(jnp.int32), critical_value


@partial(
	jax.jit,
	static_argnames=[
		"n_samples",
		"n_calibration_samples",
		"n_observations",
		"t",
		"alpha",
		"test_fn",
	],
)
def run_tests_for_configuration(
		key,
		data_ps,
		data_ws,
		null_ps,
		null_ws,
		n_samples,
		n_calibration_samples,
		n_observations,
		t,
		alpha,
		test_fn,
):
	"""
	Run one known-reference homogeneity test for a fixed p0.

	Notes
	-----
	The function generates simulations under the null hypothesis for calibration
	and simulations under the alternative hypothesis for power, then applies the
	selected test. This is the JIT-compiled simulation kernel for Figure 2(a).
	"""
	key_calibration, key_experiment, key_test = jax.random.split(key, 3)

	null_counts = sample_binomial_mixture_counts_batch(
		key=key_calibration,
		ps=null_ps,
		ws=null_ws,
		n_datasets=n_calibration_samples,
		n_observations=n_observations,
		t=t,
	)

	alt_counts = sample_binomial_mixture_counts_batch(
		key=key_experiment,
		ps=data_ps,
		ws=data_ws,
		n_datasets=n_samples,
		n_observations=n_observations,
		t=t,
	)

	test_results, critical_value = test_fn(
		key=key_test,
		null_counts=null_counts,
		alt_counts=alt_counts,
		null_ps=null_ps,
		null_ws=null_ws,
		n_observations=n_observations,
		t=t,
		alpha=alpha,
	)

	return test_results, critical_value


# Simulation parameters and Figure 2(a) experiment.
# Vary the location p0 of the point-mass null, run each test against the
# matched-mean and shifted-mean families, and plot them in separate panels.

if __name__ == "__main__":
	# Master PRNG key for reproducibility.
	key = jax.random.PRNGKey(0)

	# Number of datasets simulated under the alternative hypothesis.
	n_samples = 5000
	# Number of simulations under the null hypothesis used for calibration.
	n_calibration_samples = 2000

	# Nominal type-I error level.
	alpha = 0.05

	# Use the same sample sizes for both alternative families.
	n_observations = 300
	t = 50
	alternative_families = ["Alternative matches the mean of the null",
							"Alternative differs in mean from the null"]

	p0_values = [
		float(p0)
		for p0 in jnp.linspace(
			start=0.01,
			stop=0.5,
			num=50,
		)
	]

	tests = {
		"chi_squared_test": chi_squared_test,
		"local_minimax_test": local_minimax_test,
		"global_minimax_test": global_minimax_test,
		"plugin_test": plugin_test,
	}

	rows = []

	for family_idx, family in enumerate(alternative_families):
		family_key = jax.random.fold_in(key, family_idx)

		for test_idx, (test_name, test_fn) in enumerate(tests.items()):
			test_key = jax.random.fold_in(family_key, test_idx)

			outer_keys = jax.random.split(
				test_key,
				len(p0_values),
			)

			for key_run, p0 in tqdm(
					zip(outer_keys, p0_values),
					total=len(p0_values),
					desc=f"Running {family}: {test_name}",
			):
				null_ps, null_ws = homogeneity_null_distribution(p0=p0)

				if family == "Alternative matches the mean of the null":
					epsilon = float(
						2.0
						* jnp.sqrt(p0 * (1.0 - p0) / t)
						/ n_observations ** 0.25
					)
					alt_ps, alt_ws = two_point_alternative(
						p0=p0,
						epsilon=epsilon,
					)
				else:
					epsilon = 0.1 * p0
					alt_ps, alt_ws = mean_shifted_alternative(
						p0=p0,
						epsilon=epsilon,
					)

				test_results, _ = run_tests_for_configuration(
					key=key_run,
					data_ps=alt_ps,
					data_ws=alt_ws,
					null_ps=null_ps,
					null_ws=null_ws,
					n_samples=n_samples,
					n_calibration_samples=n_calibration_samples,
					n_observations=n_observations,
					t=t,
					alpha=alpha,
					test_fn=test_fn,
				)

				test_results = jax.device_get(test_results)

				for test_result in test_results:
					rows.append({
						"family": family,
						"test": test_name,
						"p0": float(p0),
						"test_result": int(test_result),
					})

	test_results = pd.DataFrame(rows)

	plot_df = (
		test_results
		.groupby(
			["family", "test", "p0"],
			as_index=False,
		)["test_result"]
		.mean()
		.rename(columns={"test_result": "rejection_rate"})
	)

	test_name_map = {
		"chi_squared_test": "Chi-squared",
		"local_minimax_test": "Local minimax",
		"global_minimax_test": "Global minimax",
		"plugin_test": "Plug-in",
	}

	plot_df["test"] = plot_df["test"].replace(test_name_map)
	plot_df["family"] = pd.Categorical(
		plot_df["family"],
		categories=alternative_families,
		ordered=True,
	)

	plot_df = plot_df.sort_values(
		[
			"family",
			"test",
			"p0",
		]
	)

	p_power = (
			ggplot(
				plot_df,
				aes(
					x="p0",
					y="rejection_rate",
					color="test",
					group="test",
				),
			)
			+ geom_line()
			+ geom_point(size=1.8)
			+ scale_color_manual(values={
		"Chi-squared": "#F8766D",
		"Global minimax": "#00BA38",
		"Local minimax": "#C77CFF",
		"Plug-in": "#619CFF",
	})
			+ facet_wrap("~family", nrow=1)
			+ theme_bw()
			+ labs(
		title=(
			"Power under alternative distribution"
			f" (n={n_observations}, t={t})"
		),
		x=r"Location of null distribution's mean",
		y="Power (higher is better)",
		color="Test",
	) + guides(
		color=guide_legend(nrow=1)
	)
			+ theme(
		legend_position="top",
		legend_box_just="left",
		legend_margin=0,
		legend_box_margin=0,
		plot_title=element_text(size=9, margin={"b": 2}),
		legend_title=element_text(margin={"r": 4}),
	)
	)

	p_power.save(
		"homogeneity_known_reference.pdf",
		dpi=300,
		width=8,
		height=4,
	)