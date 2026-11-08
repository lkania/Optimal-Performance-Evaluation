# Simulations for *Optimal Goodness-of-Fit Testing for Performance Evaluation*

This directory contains the Python simulation scripts used to generate the
simulation figures in Section 6 of the paper:

- `gof.py` → **Figure 1**: goodness-of-fit testing.
- `homogeneity_known_reference.py` → **Figure 2**: homogeneity testing with a known reference effect.
- `homogeneity_unkown_reference.py` → **Figure 3**: homogeneity testing
  without a known reference effect.

Each simulation saves only its figure as a PDF in the current working directory.

---

## 1. Create the Python environment

The simulation environment uses the following package versions:

| Package | Version |
| --- | --- |
| Python | 3.11.15 |
| JAX | 0.9.2 |
| jaxlib | 0.9.2 |
| NumPy | 2.4.4 |
| pandas | 3.0.3 |
| tqdm | 4.67.3 |
| SciPy | 1.17.1 |
| plotnine | 0.15.4 |
| matplotlib | 3.9.4 |

Create a Conda environment with these versions using:

```bash
conda create -n performance-evaluation -c conda-forge \
    python=3.11.15 \
    jax=0.9.2 \
    jaxlib=0.9.2 \
    numpy=2.4.4 \
    pandas=3.0.3 \
    tqdm=4.67.3 \
    scipy=1.17.1 \
    plotnine=0.15.4 \
    matplotlib=3.9.4
```

Activate the environment with:

```bash
conda activate performance-evaluation
```

---

## 2. Power estimation and calibration

In each experiment, `n` is the number of observations per dataset, and `t` is the number of trials per observation. Each observation records the number of successes in those trials.

For each simulation configuration, 5,000 independent datasets are generated under the alternative hypothesis. Empirical power is the proportion of these  datasets for which the test rejects the null hypothesis. All experiments use  a nominal significance level of 5%.

For goodness-of-fit testing and homogeneity testing with a known reference effect, the null distribution is known. Critical values are  estimated using 2,000 datasets generated under this distribution, independently of the datasets used to estimate power.

For homogeneity testing without a known reference effect, the null hypothesis specifies a common but unknown success probability. Calibration is therefore  repeated over a grid of possible success probabilities, using 2,000 datasets at each grid point. Each test uses the largest empirical 95th percentile of its test statistic across the grid as its critical value.

---

## 3. Figure 1: goodness-of-fit testing

Run:

```bash
python gof.py
```

The script simulates the goodness-of-fit problem in equation (2) under
null and alternative mixing distributions that match their first three
moments. Each simulated dataset contains `n = 1000` observations, with
`t = 50` trials per observation in the left panel and `t = 400` in the right
panel.

It saves:

```text
gof.pdf
```

The figure compares the following tests:

- **Plug-in test — equation (10).**
- **Pearson chi-squared test.**
- **Global minimax test — equation (15).**

The x-axis is the 1-Wasserstein distance between the null and alternative
mixing distributions, and the y-axis is empirical power.

---

## 4. Figure 2: homogeneity with a known reference effect

Run:

```bash
python homogeneity_known_reference.py
```

The script simulates the homogeneity testing problem with known reference effect in equation (19). Each simulated dataset contains `n = 300` observations, with `t = 50` trials per observation. The experiment is repeated at 50 reference locations between 0.01 and 0.5. The left panel uses two-point alternatives that match the mean of the null distribution; the right panel uses two-point alternatives that differ in mean.

It saves:

```text
homogeneity_known_reference.pdf
```

The figure compares the following tests:

- **Plug-in test — equation (10).**
- **Pearson chi-squared test.**
- **Local minimax test — equation (23).**
- **Global minimax test — equation (15).**

The x-axis is the location of the null point mass, and the y-axis is empirical power.

---

## 5. Figure 3: homogeneity without a known reference effect

Run:

```bash
python homogeneity_unkown_reference.py
```

The script simulates the homogeneity testing problem with unknown reference effect in equation (25). Each simulated dataset contains `n = 300` observations, with `t = 2` trials per observation in the left panel and `t = 10` in the right panel. The alternatives are symmetric two-point distributions with varying means.

It saves:

```text
homogeneity_unknown_reference.pdf
```

The figure compares the following tests:

- **Cochran chi-squared test.**
- **Debiased Cochran chi-squared test — equation (27).**

The x-axis is the mean of the alternative distribution, and the y-axis is empirical power. The dashed line marks the denominator-truncation threshold for the debiased Cochran test.