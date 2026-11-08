# Simulation for Figure 2(b).
from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pandas as pd

from shared_utilities import (
	two_point_alternative,
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
	geom_vline,
	facet_wrap,
	theme_bw,
	theme,
	labs,
	scale_color_manual,
	scale_linetype_manual,
)


def kernel_variance(
		x,
		y,
		tx,
		ty,
):
	"""
	Evaluate the pairwise kernel of the unbiased variance estimator.

	Notes
	-----
	Equation (30) defines V_hat as a second-order U-statistic. This function
	evaluates its kernel h(x, y) for arbitrary trial counts tx and ty.
	"""
	return (
			x * (x - 1.0) / (tx * (tx - 1.0))
			+ y * (y - 1.0) / (ty * (ty - 1.0))
			- 2.0 * x * y / (tx * ty)
	)


def kernel_mu_m1(
		x,
		y,
		tx,
		ty,
):
	"""
	Evaluate the pairwise kernel of the unbiased normalizing estimator.

	Notes
	-----
	Equation (33) normalizes V_hat by an unbiased estimator mu_hat of
	m_1(pi)(1-m_1(pi)). This function evaluates the pairwise kernel used to form
	that estimator.
	"""
	return (
			x / tx
			+ y / ty
			- 2.0 * x * y / (tx * ty)
	)


def estimate_variance(
		counts,
		n_observations,
		t,
):
	"""
	Compute the unbiased variance estimator V_hat from Equation (30).

	Notes
	-----
	The fingerprint counts represent all ordered observation pairs. When two
	distinct observations have the same fingerprint value x, there are
	count_x * (count_x - 1) such pairs. Subtracting diag(counts) from the outer
	product of counts preserves those pairs while removing only self-pairs.
	"""
	counts = counts.reshape(-1).astype(jnp.float64)

	positions = jnp.arange(
		start=0,
		stop=t + 1,
		step=1,
		dtype=jnp.float64,
	)

	kernel_matrix = kernel_variance(
		x=positions.reshape(-1, 1),
		y=positions.reshape(1, -1),
		tx=t,
		ty=t,
	)

	weights = (
			counts.reshape(-1, 1)
			* counts.reshape(1, -1)
			- jnp.diag(counts)
	)

	kernel_matrix = kernel_matrix * weights

	return jnp.sum(kernel_matrix) / (
			2.0 * n_observations * (n_observations - 1.0)
	)


def estimate_mu_m1(
		counts,
		n_observations,
		t,
):
	"""
	Compute the unbiased normalizing estimator mu_hat used in Equation (33).

	Notes
	-----
	The same observations used in V_hat are reused here. The histogram weights
	count all distinct observation pairs, including pairs with equal fingerprint
	values, and the resulting kernel average uses the U-statistic normalization.
	"""
	counts = counts.reshape(-1).astype(jnp.float64)

	positions = jnp.arange(
		start=0,
		stop=t + 1,
		step=1,
		dtype=jnp.float64,
	)

	kernel_matrix = kernel_mu_m1(
		x=positions.reshape(-1, 1),
		y=positions.reshape(1, -1),
		tx=t,
		ty=t,
	)

	weights = (
			counts.reshape(-1, 1)
			* counts.reshape(1, -1)
			- jnp.diag(counts)
	)

	kernel_matrix = kernel_matrix * weights

	return jnp.sum(kernel_matrix) / (
			2.0 * n_observations * (n_observations - 1.0)
	)


def debiased_cochran_statistic(
		counts,
		n_observations,
		t,
		gamma,
):
	"""
	Compute the debiased Cochran statistic R(X) in Equation (33).

	Notes
	-----
	The statistic is V_hat / max(mu_hat, gamma). The numerator is Equation (30),
	the denominator uses the unbiased estimator mu_hat, and gamma is set to
	min(1/2, 5/(alpha*n*t)) when the batch statistics are computed.
	"""
	v_hat = estimate_variance(
		counts=counts,
		n_observations=n_observations,
		t=t,
	)

	mu_hat = estimate_mu_m1(
		counts=counts,
		n_observations=n_observations,
		t=t,
	)

	return v_hat / jnp.maximum(
		mu_hat,
		gamma,
	)


def cochran_statistic(
		counts,
		n_observations,
		t,
):
	"""
	Compute the classical Cochran chi-squared comparison statistic.

	Notes
	-----
	The pooled mean estimates the common success probability. The sample variance
	of X_i/t is normalized by the plug-in Bernoulli variance. This is the Cochran
	baseline shown in Figure 2(b).
	"""
	counts = counts.reshape(-1).astype(jnp.float64)

	positions = (
			jnp.arange(
				start=0,
				stop=t + 1,
				step=1,
				dtype=jnp.float64,
			)
			/ t
	)

	mean = jnp.sum(positions * counts) / n_observations
	bernoulli_variance = mean * (1.0 - mean)

	chi2 = (
			jnp.sum(
				(positions - mean) ** 2 * counts
			)
			/ (n_observations - 1.0)
	)

	return jnp.where(
		bernoulli_variance > 0.0,
		t * chi2 / bernoulli_variance,
		0.0,
	)


def calibrate_max_critical_value(
		key,
		p0_values,
		n_calibration_samples,
		n_observations,
		t,
		alpha,
		test_statistic_fn,
):
	"""
	Calibrate one test over a grid of point-mass null distributions.

	Notes
	-----
	Equation (33) uses the supremum of the null quantile over pi in S. The
	simulation approximates this supremum on the supplied p0 grid, computes a
	(1-alpha)-quantile at each value, and takes the maximum.
	"""
	keys = jax.random.split(
		key,
		len(p0_values),
	)

	critical_values = []

	for key_p0, p0 in zip(keys, p0_values):
		null_ps, null_ws = homogeneity_null_distribution(
			p0=p0,
		)

		null_counts = sample_binomial_mixture_counts_batch(
			key=key_p0,
			ps=null_ps,
			ws=null_ws,
			n_datasets=n_calibration_samples,
			n_observations=n_observations,
			t=t,
		)

		calibration_statistics = batch_test_statistics(
			counts=null_counts,
			test_statistic_fn=test_statistic_fn,
			n_observations=n_observations,
			t=t,
		)

		critical_values.append(
			upper_quantile(
				values=calibration_statistics,
				alpha=alpha,
			)
		)

	critical_values = jnp.array(critical_values)

	return jnp.max(critical_values), critical_values


@partial(
	jax.jit,
	static_argnames=[
		"n_samples",
		"n_observations",
		"t",
		"test_statistic_fn",
	],
)
def run_tests_for_configuration(
		key,
		data_ps,
		data_ws,
		n_samples,
		n_observations,
		t,
		critical_value,
		test_statistic_fn,
):
	"""
	Run one unknown-reference homogeneity test at a fixed alternative.

	Notes
	-----
	The critical value is calibrated before this function is called. This
	JIT-compiled function generates datasets under the alternative hypothesis,
	computes the selected statistic, and returns its rejection indicators.
	"""
	alt_counts = sample_binomial_mixture_counts_batch(
		key=key,
		ps=data_ps,
		ws=data_ws,
		n_datasets=n_samples,
		n_observations=n_observations,
		t=t,
	)

	test_statistics = batch_test_statistics(
		counts=alt_counts,
		test_statistic_fn=test_statistic_fn,
		n_observations=n_observations,
		t=t,
	)

	test_results = test_statistics >= critical_value

	return test_results.astype(jnp.int32), test_statistics


# Simulation parameters and Figure 2(b) experiment.
# Calibrate each test separately for each (n, t), then plot power versus the
# alternative mean p0 in two panels, as in the goodness-of-fit experiment.

if __name__ == "__main__":
	# Master PRNG key for reproducibility.
	key = jax.random.PRNGKey(0)

	# Number of datasets simulated under the alternative hypothesis.
	n_samples = 5_000
	# Number of simulations under each point-mass calibration null.
	n_calibration_samples = 2_000

	alpha = 0.05

	sample_configurations = [
		{
			"n_observations": 300,
			"t": 2,
		},
		{
			"n_observations": 300,
			"t": 10,
		},
	]

	# Construct a supported p0 grid for each panel. For p0 <= 1/2,
	# epsilon <= p0 requires p0 >= 4 / (t*sqrt(n_observations) + 4).
	for sample_config in sample_configurations:
		n_observations = sample_config["n_observations"]
		t = sample_config["t"]
		p0_lower_bound = float(
			4.0 / (t * jnp.sqrt(n_observations) + 4.0)
		)
		if p0_lower_bound >= 0.5:
			raise ValueError(
				"Increase n_observations or t to obtain a supported p0 range."
			)
		p0_start = p0_lower_bound + 1e-6 * (0.5 - p0_lower_bound)
		sample_config["p0_values"] = [
			float(p0)
			for p0 in jnp.linspace(p0_start, 0.5, 50)
		]

	# Symmetry around 1/2 lets [0, 0.5] cover the composite null.
	calibration_p0_values = jnp.linspace(
		0.0,
		0.5,
		200,
	)

	tests = {
		"debiased_cochran_test": debiased_cochran_statistic,
		"cochran_test": cochran_statistic,
	}

	rows = []

	for test_idx, (test_name, base_statistic_fn) in enumerate(tests.items()):
		test_key = jax.random.fold_in(key, test_idx)

		for config_idx, sample_config in enumerate(sample_configurations):
			n_observations = sample_config["n_observations"]
			t = sample_config["t"]
			p0_values = sample_config["p0_values"]
			panel = f"n = {n_observations}, t = {t}"

			# Both gamma and the composite-null critical value depend on (n, t).
			gamma = float(
				jnp.minimum(
					0.5,
					5.0 / (alpha * n_observations * t),
				)
			)
			if test_name == "debiased_cochran_test":
				test_statistic_fn = partial(base_statistic_fn, gamma=gamma)
			else:
				test_statistic_fn = base_statistic_fn

			config_key = jax.random.fold_in(test_key, config_idx)
			key_calibration, key_experiment = jax.random.split(config_key, 2)

			critical_value, _ = calibrate_max_critical_value(
				key=key_calibration,
				p0_values=calibration_p0_values,
				n_calibration_samples=n_calibration_samples,
				n_observations=n_observations,
				t=t,
				alpha=alpha,
				test_statistic_fn=test_statistic_fn,
			)

			outer_keys = jax.random.split(
				key_experiment,
				len(p0_values),
			)

			for key_run, p0 in tqdm(
					zip(outer_keys, p0_values),
					total=len(p0_values),
					desc=f"Running {test_name}: {panel}",
			):
				epsilon = float(
					2.0
					* jnp.sqrt(p0 * (1.0 - p0) / t)
					/ n_observations ** 0.25
				)
				alt_ps, alt_ws = two_point_alternative(
					p0=p0,
					epsilon=epsilon,
				)

				test_results, _ = run_tests_for_configuration(
					key=key_run,
					data_ps=alt_ps,
					data_ws=alt_ws,
					n_samples=n_samples,
					n_observations=n_observations,
					t=t,
					critical_value=critical_value,
					test_statistic_fn=test_statistic_fn,
				)

				test_results = jax.device_get(test_results)

				for test_result in test_results:
					rows.append({
						"test": test_name,
						"panel": panel,
						"p0": float(p0),
						"test_result": int(test_result),
					})

	test_results = pd.DataFrame(rows)

	plot_df = (
		test_results
		.groupby(
			["test", "panel", "p0"],
			as_index=False,
		)["test_result"]
		.mean()
		.rename(columns={"test_result": "rejection_rate"})
	)

	test_name_map = {
		"debiased_cochran_test": "Debiased Cochran",
		"cochran_test": "Cochran",
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
			"p0",
		]
	)

	# Mark p0 * (1 - p0) = gamma using the root in [0, 0.5].
	threshold_rows = []
	for sample_config, panel in zip(sample_configurations, panel_order):
		gamma = min(
			0.5,
			5.0 / (
					alpha * sample_config["n_observations"] * sample_config["t"]
			),
		)
		if gamma <= 0.25:
			threshold_rows.append({
				"experiment_panel": panel,
				"p0_threshold": float(
					(1.0 - jnp.sqrt(1.0 - 4.0 * gamma)) / 2.0
				),
			})
	threshold_df = pd.DataFrame(
		threshold_rows,
		columns=["experiment_panel", "p0_threshold"],
	)
	threshold_df["experiment_panel"] = pd.Categorical(
		threshold_df["experiment_panel"],
		categories=panel_order,
		ordered=True,
	)
	threshold_df["Threshold"] = r"$\text{mean}\cdot (1-\text{mean})=\gamma$"

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
			+ geom_vline(
		data=threshold_df,
		mapping=aes(
			xintercept="p0_threshold",
			linetype="Threshold",
		),
		color="black",
		inherit_aes=False,
		show_legend=True,
	)
			+ geom_line()
			+ geom_point(size=2)
			+ scale_color_manual(values={
		"Cochran": "#F8766D",
		"Debiased Cochran": "#00BFC4",
	})
			+ scale_linetype_manual(
		name="",
		values={r"$\text{mean}\cdot (1-\text{mean})=\gamma$": "dashed"},
	)
			+ facet_wrap("~ experiment_panel", nrow=1, scales="free_x")
			+ theme_bw()
			+ labs(
		title="Power under alternative distribution",
		x=r"Location of alternative distribution's mean",
		y="Power (higher is better)",
		color="Test",
	)
			+ theme(
		legend_position="top",
		legend_box="horizontal",
		legend_margin=0,
		legend_box_margin=0,
		panel_spacing_x=0.03,
		plot_title=element_text(margin={"b": 2}),
		legend_title=element_text(margin={"r": 4}),
	)
	)

	p_power.save(
		"homogeneity_unknown_reference.pdf",
		dpi=300,
		width=8,
		height=4,
	)