## Statistics refresher

Collection of useful statistics concepts that might be useful to understand the models and techniques employed in this package.

----
### Hypothesis testing

Hypothesis testing is a statistical technique employed to gain evidence from available data that a specific hypothesis (the *null hypothesis*) is true, unless the data suggests it is not (hence, we accept the *alternative hypothesis*).

More formally, suppose we have a specific parameter space $\Theta$, and we partition it into two sets such that

$$
H_{0}: \theta \in \Theta_{0}
\quad \text{versus} \quad
H_{1}: \theta \in \Theta_{1}
$$

We test an hypothesis by defining a so-called *rejection region*, such that if data provides a specific test statistic that falls into that region, we reject the null hypothesis.

The rejection region is determined by a significance level $\alpha$ (commonly $\alpha = 0.05$). Intuitively:

- if the p-value is smaller than $\alpha$, we reject the null hypothesis;
- if the p-value is greater than $\alpha$, we fail to reject the null hypothesis.

Importantly, *failing to reject the null hypothesis does not imply that the null hypothesis is true*. It only means that the available data does not provide sufficient statistical evidence against it.

Similarly, rejecting the null hypothesis does not "prove" that the alternative hypothesis is true, but rather that the observed data is statistically unlikely under the null hypothesis assumptions.

Therefore, hypothesis testing should be interpreted as an evidence-based probabilistic framework rather than a proof-based deterministic one.

----
### Augmented Dickey-Fuller (ADF) Test

The ADF test is used to assess whether a time series is stationary by checking for the presence of a *unit root*.

It is commonly employed before fitting models such as:
- ARIMA,
- SARIMAX,
- autoregressive statistical models.

The hypotheses are:

$$
H_0: \text{the time series is non-stationary}
$$

$$
H_1: \text{the time series is stationary}
$$

Interpretation:
- small p-value $\Rightarrow$ reject $H_0$ $\Rightarrow$ evidence of stationarity;
- large p-value $\Rightarrow$ fail to reject $H_0$ $\Rightarrow$ insufficient evidence for stationarity.

The test may become unreliable with:
- small datasets,
- strong seasonality,
- structural breaks,
- poor lag selection.

In forecasting systems, ADF is mainly used as a diagnostic tool to determine whether preprocessing steps such as differencing may be required before fitting ARIMA-like models.

---

### KPSS Test

The KPSS test is another stationarity test, but with the opposite null hypothesis compared to ADF.

It is typically used together with ADF to obtain a more robust understanding of the statistical properties of the series.

The hypotheses are:

$$
H_0: \text{the time series is stationary}
$$

$$
H_1: \text{the time series is non-stationary}
$$

Interpretation:
- small p-value $\Rightarrow$ reject $H_0$ $\Rightarrow$ evidence of non-stationarity;
- large p-value $\Rightarrow$ fail to reject $H_0$ $\Rightarrow$ data compatible with stationarity.

The test may fail or become unstable in the presence of:
- trends,
- strong seasonality,
- small datasets,
- structural breaks.

In practice, KPSS complements the ADF test and helps validate assumptions required by statistical forecasting models.