# Statistics refresher

Collection of useful statistics concepts that might be useful to understand the models and techniques employed in this package.

----
## Hypothesis testing

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

---

## Autocorrelation 

Autocorrelation measures the linear relationship between lagged values of a time series, namely between $y_t$ and $y_{t-k}$.

In other words, it quantifies how strongly past observations influence future observations at a given lag $k$.

The autocorrelation of a time series $y$ at lag $k$ is defined as:

$$
\frac{
\sum_{t = k + 1}^{T}
(y_{t-k} - \bar{y})(y_t - \bar{y})
}{
\sum_{t = 1}^{T}
(y_t - \bar{y})^2
}
$$

In time series forecasting, autocorrelation is an important diagnostic tool that helps identify:
- seasonality,
- persistence,
- trend-related temporal dependence,
- decay patterns.

Seasonality can often be identified from the ACF plot when:
- significant spikes appear at regular lags,
- especially at multiples of a seasonal period $T$.

For example:
- daily data with weekly seasonality often shows spikes at lags 7, 14, 21, ...

A slowly decaying ACF instead is often associated with trend or non-stationarity.

---

### Autocorrelation Function (ACF)

The *Autocorrelation Function* (ACF) measures the autocorrelation of a time series across multiple lags.

An ACF plot typically has:
- x-axis: lag values,
- y-axis: autocorrelation coefficient.

Interpretation:
- periodic spikes often indicate seasonality;
- slow decay often indicates trend or non-stationarity;
- values close to zero indicate weak temporal dependence.

The ACF is commonly used in:
- exploratory data analysis,
- seasonality detection,
- ARIMA diagnostics.

---
## Partial autocorrelation

Partial autocorrelation measures the direct relationship between $y_t$ and $y_{t-k}$ after removing the effect of intermediate lags.

While autocorrelation includes both:
- direct effects,
- indirect effects propagated through intermediate lags,

partial autocorrelation isolates only the direct contribution of lag $k$.

---

### Partial Autocorrelation Function (PACF)

The *Partial Autocorrelation Function* (PACF) measures partial autocorrelation across multiple lags.

A PACF plot is particularly useful for identifying:
- important direct lag dependencies,
- autoregressive structure,
- the order $p$ in ARIMA models.

Interpretation:
- strong spikes at early lags indicate direct temporal dependence;
- a sharp cutoff after lag $p$ often suggests an AR($p$) process.

In practice:
- ACF is mainly used to analyze overall temporal structure and seasonality;
- PACF is mainly used to identify direct lag effects in autoregressive models.

---

## Decomposition Plots

Decomposition plots are diagnostic tools used to separate a time series into its main components:

```text
Observed Series
=
Trend
+ Seasonal Component
+ Residual Noise
```

or, for multiplicative models:

```text
Observed Series
=
Trend × Seasonal Component × Residual Noise
```

A decomposition plot typically contains:
- observed series,
- trend,
- seasonality,
- residuals.

These plots help identify:
- long-term trends,
- seasonal patterns,
- remaining unexplained noise.

---

### Additive vs Multiplicative Models

#### Additive decomposition

Assumes constant seasonal fluctuations:

```text
series = trend + seasonality + residual
```

Preferred when:
- variance is approximately constant,
- seasonal amplitude remains stable over time.

---

#### Multiplicative decomposition

Assumes seasonal fluctuations scale with the series level:

```text
series = trend × seasonality × residual
```

Preferred when:
- variance increases with the trend,
- seasonal oscillations become larger over time.

---

### Seasonal Period Selection

The seasonal period is usually inferred through:
- domain knowledge,
- ACF/PACF analysis,
- visual inspection.

Example:
- daily data with weekly seasonality often uses:

```python
period = 7
```

Periodic ACF spikes at:
- 7,
- 14,
- 21,
- ...

often suggest weekly seasonality.

---

## Residual Analysis

Residuals represent the unexplained component of the model:

```text
residual = observed - predicted
```

Ideally, residuals should resemble:
- white noise,
- zero-centered random fluctuations,
- no remaining temporal structure.

---

### Residual Histogram

Used to visually inspect:
- symmetry,
- skewness,
- approximate Gaussian shape.

Ideally:
- residuals should be centered around zero,
- distribution should appear roughly bell-shaped.

---

### QQ Plot

A QQ plot compares residual quantiles against a theoretical Gaussian distribution.

Interpretation:
- points close to the diagonal line suggest approximate normality;
- large deviations suggest skewness or heavy tails.

---

### Residual Autocorrelation

Residual ACF plots verify whether temporal structure remains after modeling.

Ideally:
- most autocorrelations should lie within confidence bands.

Strong residual autocorrelation may indicate that:
- trend,
- seasonality,
- or other temporal dependencies

were not fully captured by the model.

In practice, approximate white-noise residuals are usually more important than perfect Gaussianity.