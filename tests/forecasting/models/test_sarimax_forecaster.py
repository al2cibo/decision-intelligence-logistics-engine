import numpy as np
import polars as pl
import pytest
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecasting.models.sarimax_forecaster import SARIMAXForecaster


def _trending_series(n: int) -> list[float]:
    return [10.0 + 2.0 * i for i in range(n)]


class TestSARIMAXForecaster:
    def test_predict_before_fit_raises(self):
        forecaster = SARIMAXForecaster()
        with pytest.raises(
            RuntimeError, match=r"fit\(\) must be called before predict\(\)"
        ):
            forecaster.predict(pl.DataFrame({"demand": [1.0, 2.0]}))

    def test_fit_missing_column_raises(self):
        forecaster = SARIMAXForecaster(target_col="demand")
        with pytest.raises(ValueError, match="Column 'demand' not found"):
            forecaster.fit(pl.DataFrame({"other": [1.0, 2.0]}))

    def test_predict_missing_column_raises(self):
        forecaster = SARIMAXForecaster(target_col="demand")
        forecaster.fit(pl.DataFrame({"demand": [1.0, 2.0, 3.0, 4.0, 5.0]}))
        with pytest.raises(ValueError, match="Column 'demand' not found"):
            forecaster.predict(pl.DataFrame({"other": [1.0]}))

    def test_no_nulls_and_correct_length(self):
        train = pl.DataFrame({"demand": _trending_series(60)})
        test = pl.DataFrame({"demand": [0.0] * 10})
        forecaster = SARIMAXForecaster(order=(1, 1, 0))
        forecaster.fit(train)
        result = forecaster.predict(test)
        assert result.height == 10
        assert result["sarimax_forecast"].is_null().sum() == 0

    def test_forecast_is_genuinely_out_of_sample(self):
        """Regression test for the fittedvalues[:n] bug.

        predict() must extrapolate beyond the end of training via
        forecast(steps=n), not replay the model's in-sample fit for the
        first n training rows (which is what the old buggy branch did
        whenever len(df) <= train_length — i.e. on every real call).
        """
        train_values = _trending_series(60)  # 10, 12, ..., 128
        train = pl.DataFrame({"demand": train_values})
        test = pl.DataFrame({"demand": [0.0] * 10})

        forecaster = SARIMAXForecaster(order=(1, 1, 0))
        forecaster.fit(train)
        result = forecaster.predict(test)
        forecast_values = result["sarimax_forecast"].to_list()

        # Ground truth: fit the same series independently and forecast forward.
        independent_fit = SARIMAX(
            np.array(train_values), order=(1, 1, 0), seasonal_order=(0, 0, 0, 0)
        ).fit(disp=False)
        expected = independent_fit.forecast(steps=10)

        np.testing.assert_allclose(forecast_values, expected, rtol=1e-6)

        # The buggy implementation returned fittedvalues[:10], which sit near
        # the start of the training trend (~10-30). Genuine out-of-sample
        # forecasts must continue the trend beyond the end of training (128).
        assert min(forecast_values) > train_values[-1]
