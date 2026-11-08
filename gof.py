# Simulation for Figure 1.
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
	sample_binomial_mixture_counts_batch,
	wasserstein_1d,
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
	scale_color_manual
)


def chebyshev_nodes(lower, upper, n_nodes):
	"""
	Construct Chebyshev nodes on a specified interval.

	Notes
	-----
	The nodes are used to build the moment-matching distributions that generate
	Figure 1. These pairs satisfy the moment constraints appearing in Equation (19).
	"""
	i = jnp.arange(n_nodes)

	nodes = (
			0.5 * (lower + upper)
			+ 0.5 * (upper - lower)
			* jnp.cos((2 * i + 1) / (2 * n_nodes) * jnp.pi)
	)

	return nodes


def derivative_at_roots(roots):
	"""
	Evaluate the derivative of the monic root polynomial at each root.

	Notes
	-----
	For roots r_i, the returned value is prod_{j != i}(r_i-r_j). Its sign separates
	the two moment-matching distributions and its magnitude determines their
	weights.
	"""
	roots = roots.reshape(-1)

	pairwise_differences = roots[:, None] - roots
	stabilized_differences = pairwise_differences + jnp.eye(roots.shape[0])

	derivative_values = jnp.prod(
		stabilized_differences,
		axis=1,
	)

	return derivative_values


def raw_moments(ps, ws, n_moments):
	"""
	Compute the first raw moments of a discrete mixing distribution.

	Notes
	-----
	The l-th moment is m_l(pi) = E_{p~pi}[p^l]. Equality of these moments is the
	constraint used in Equation (19).
	"""
	powers = jnp.arange(
		start=1,
		stop=n_moments + 1,
		step=1,
	)

	moments = jnp.sum(
		ws[:, None] * ps[:, None] ** powers[None, :],
		axis=0,
	)

	return moments


def moment_matching_arrays(lower, upper, n_moments, tol):
	"""
	Construct two discrete mixing distributions with matching low-order moments.

	Notes
	-----
	Chebyshev nodes and alternating derivative signs generate the two distributions.
	The function verifies the moment constraints in Equation (19) up to ``tol``.
	These distributions are used to generate the adversarial pairs in Figure 1.
	"""
	roots = chebyshev_nodes(
		lower=lower,
		upper=upper,
		n_nodes=n_moments + 2,
	)

	derivative_values = derivative_at_roots(roots)

	positive_mask = derivative_values > 0
	negative_mask = derivative_values < 0

	ps_null = roots[positive_mask]
	ws_null = 1.0 / derivative_values[positive_mask]
	ws_null = ws_null / jnp.sum(ws_null)

	ps_alt = roots[negative_mask]
	ws_alt = 1.0 / jnp.abs(derivative_values[negative_mask])
	ws_alt = ws_alt / jnp.sum(ws_alt)

	assert abs(ps_null.shape[0] - ps_alt.shape[0]) <= 1

	null_moments = raw_moments(
		ps=ps_null,
		ws=ws_null,
		n_moments=n_moments,
	)

	alt_moments = raw_moments(
		ps=ps_alt,
		ws=ws_alt,
		n_moments=n_moments,
	)

	moment_diff = float(jnp.max(jnp.abs(null_moments - alt_moments)))

	assert moment_diff <= tol, (
		f"Moment matching failed: "
		f"lower={lower}, upper={upper}, "
		f"n_moments={n_moments}, "
		f"moment_diff={moment_diff}, tol={tol}"
	)

	return ps_null, ws_null, ps_alt, ws_alt


def pad_distribution(ps, ws, max_support_size):
	"""
	Pad a discrete distribution with zero-weight support points.

	Notes
	-----
	Padding leaves the distribution unchanged while keeping JAX array shapes fixed
	across different numbers of support points.
	"""
	n_pad = max_support_size - ps.shape[0]

	if n_pad == 0:
		return ps, ws

	ps_pad = jnp.full(
		shape=(n_pad,),
		fill_value=ps[0],
		dtype=ps.dtype,
	)

	ws_pad = jnp.zeros(
		shape=(n_pad,),
		dtype=ws.dtype,
	)

	ps = jnp.concatenate((ps, ps_pad))
	ws = jnp.concatenate((ws, ws_pad))

	return ps, ws


@partial(
	jax.jit,
	static_argnames=[
		"n_samples",
		"n_calibration_samples",
		"n_observations",
		"t",
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
	Run one goodness-of-fit test for a fixed simulation configuration.

	Notes
	-----
	The function generates simulations under the null hypothesis for calibration
	and simulations under the alternative hypothesis for power, then applies the
	selected test. This is the JIT-compiled simulation kernel used for Figure 1.
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


# Simulation parameters and Figure 1 experiment.
# For each (n, t) configuration, fix the number of matched moments and vary
# delta to obtain a denser grid of Wasserstein separations.

if __name__ == "__main__":
	# Master PRNG key for reproducibility.
	key = jax.random.PRNGKey(0)

	# Number of datasets simulated under the alternative hypothesis.
	n_samples = 5000
	# Number of simulations under the null hypothesis used for calibration.
	n_calibration_samples = 2000

	# Nominal type-I error level.
	alpha = 0.05

	tol = 1e-8

	# Both panels match the same number of moments. Their delta grids target the
	# respective power-transition regions and avoid floor/ceiling saturation.

	sample_configurations = [
		{
			"n_observations": 1000,
			"t": 50,
			"n_moments": 3,
			"delta_values": [
				float(delta)
				for delta in jnp.linspace(0.1, 0.15, 30)
			],
		},
		{
			"n_observations": 1000,
			"t": 400,
			"n_moments": 3,
			"delta_values": [
				float(delta)
				for delta in jnp.linspace(0.01, 0.06, 30)
			],
		},
	]

	max_support_size = max(
		(config["n_moments"] + 3) // 2
		for config in sample_configurations
	)

	tests = {
		"plugin_test": plugin_test,
		"chi_squared_test": chi_squared_test,
		"global_minimax_test": global_minimax_test,
	}

	rows = []

	for test_idx, (test_name, test_fn) in enumerate(tests.items()):
		configurations = [
			(sample_config, delta)
			for sample_config in sample_configurations
			for delta in sample_config["delta_values"]
		]

		test_key = jax.random.fold_in(key, test_idx)

		outer_keys = jax.random.split(
			test_key,
			len(configurations),
		)

		for key_run, config in tqdm(
				zip(outer_keys, configurations),
				total=len(configurations),
				desc=f"Running {test_name}",
		):
			sample_config, delta = config

			n_observations = sample_config["n_observations"]
			t = sample_config["t"]

			panel = f"n = {n_observations}, t = {t}"

			n_moments = int(sample_config["n_moments"])

			ps_null, ws_null, ps_alt, ws_alt = moment_matching_arrays(
				lower=0.5 - delta,
				upper=0.5 + delta,
				n_moments=n_moments,
				tol=tol,
			)

			ps_null, ws_null = pad_distribution(
				ps=ps_null,
				ws=ws_null,
				max_support_size=max_support_size,
			)

			ps_alt, ws_alt = pad_distribution(
				ps=ps_alt,
				ws=ws_alt,
				max_support_size=max_support_size,
			)

			w1_null_alt = wasserstein_1d(
				u_values=ps_null,
				u_weights=ws_null,
				v_values=ps_alt,
				v_weights=ws_alt,
			)

			test_results, _ = run_tests_for_configuration(
				key=key_run,
				data_ps=ps_alt,
				data_ws=ws_alt,
				null_ps=ps_null,
				null_ws=ws_null,
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
					"test": test_name,
					"panel": panel,
					"w1_null_alt": float(w1_null_alt),
					"test_result": int(test_result),
				})

	test_results = pd.DataFrame(rows)

	plot_df = (
		test_results
		.groupby(
			["test", "panel", "w1_null_alt"],
			as_index=False,
		)["test_result"]
		.mean()
		.rename(columns={"test_result": "rejection_rate"})
	)

	test_name_map = {
		"plugin_test": "Plug-in",
		"chi_squared_test": "Chi-squared",
		"global_minimax_test": "Global minimax",
	}

	plot_df["test"] = plot_df["test"].replace(test_name_map)

	panel_order = [
		f"n = {config['n_observations']}, t = {config['t']}"
		for config in sample_configurations
	]

	plot_df["experiment_panel"] = pd.Categorical(
		plot_df["panel"],
		categories=panel_order,
		ordered=True,
	)

	plot_df = plot_df.sort_values(
		[
			"experiment_panel",
			"test",
			"w1_null_alt",
		]
	)

	p_power = (
			ggplot(
				plot_df,
				aes(
					x="w1_null_alt",
					y="rejection_rate",
					color="test",
					group="test",
				),
			)
			+ geom_line()
			+ geom_point(size=2)
			+ scale_color_manual(values={
		"Chi-squared": "#F8766D",
		"Global minimax": "#00BA38",
		"Plug-in": "#619CFF",
	})
			+ facet_wrap("~ experiment_panel", scales="free_x")
			+ theme_bw()
			+ labs(
		title="Power under worst-case moment-matching mixing distributions",
		x="Wasserstein distance between null and alternative distributions",
		y="Power (higher is better)",
		color="Test",
	)
			+ theme(
		legend_position="top",
		legend_margin=0,
		legend_box_margin=0,
		panel_spacing_x=0.03,
		plot_title=element_text(margin={"b": 2}),
		legend_title=element_text(margin={"r": 4}),
	)
	)

	p_power.save(
		"gof.pdf",
		dpi=300,
		width=8,
		height=4,
	)