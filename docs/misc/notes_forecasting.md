# Forecasting Workflow

## 1. Problem Framing

Define:
- what to forecast
- forecast horizon
- aggregation level
- downstream decision use case

Example:
- daily destination demand forecasting for logistics allocation

---

## 2. Exploratory Data Analysis (EDA)

Inspect:
- missing values
- outliers
- duplicates
- aggregation consistency

Plot the time series to identify:
- trend
- seasonality
- volatility
- changepoints
- intermittency

---

## 3. Statistical Diagnostics

Focus mainly on:
- stationarity
- autocorrelation
- seasonality
- variance stability

Useful tools:
- ADF / KPSS tests
- ACF / PACF plots
- decomposition plots

Avoid over-focusing on probability distributions.

---

## 4. Establish Baselines First

Always compare against simple models:
- Naive
- Seasonal Naive
- Rolling Mean

Complex models should outperform these baselines.

---

## 5. Train / Validation / Test Split

Use chronological splits only:

```text
past → future
```

Prefer:
- train
- validation
- test

Even better:
- rolling / walk-forward validation

---

## 6. Fit Forecasting Models

Examples:
- ETS
- ARIMA / SARIMAX
- Prophet

---

## 7. Forecast Visualization

Plot:
- actual vs forecast
- prediction intervals
- residuals

---

## 8. Evaluate Forecast Accuracy

Common metrics:
- MAE
- RMSE
- WAPE
- MAPE (careful with zeros/small values)

---

## 9. Residual Analysis

Check for:
- systematic errors
- autocorrelation
- forecast bias
- heteroskedasticity

---

## 10. Decision-Level Evaluation

In decision intelligence systems:

```text
forecast
→ optimization
→ simulation
→ business KPIs
```

The statistically best forecast is not always the operationally best one.

---

# Recommended Workflow

## In Jupyter Notebooks

Use for:
- EDA
- experiments
- diagnostics
- visualization

Example structure:

```text
notebooks/
├── 01_eda.ipynb
├── 02_ets_experiments.ipynb
├── 03_arima_experiments.ipynb
└── 04_residual_analysis.ipynb
```


# Ideal Development Loop

```text
EDA notebook
→ understand data
→ test models manually
→ validate intuition
→ productionize into src/
→ add tests
→ integrate into pipeline
```

### To-Do:
1. Complete the notebooks, learn the data and the tech stack of sklearn + pandas 
2. Study the theory behind the ETS Forecaster model and the SARIMAX
3. Implement them and make sure the code runs properly
4. Merge the predictions for different nodes.