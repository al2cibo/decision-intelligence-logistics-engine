import polars as pl
import pytest

from forecasting.models.seasonal_forecaster import SeasonalForecaster


class TestSeasonalForecaster:

    def test_seasonal_forecaster(self):
        valid_df = pl.DataFrame(
            {"demand": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]},
        )

        lag_value = 7
        seasonal_forecaster = SeasonalForecaster(lag_value=lag_value)
        seasonal_forecaster.fit(valid_df)
        forecasted_df = seasonal_forecaster.predict(valid_df)

        assert forecasted_df.shape == (15, 2)
        # Row 7 (0-indexed): lag-7 of test row 7 = test row 0 = 1
        assert forecasted_df[7, f"seasonal_forecast_lag_{lag_value}"] == 1

    def test_no_nulls_in_forecast(self):
        train_df = pl.DataFrame({"demand": list(range(1, 15))})
        test_df = pl.DataFrame({"demand": list(range(15, 25))})
        forecaster = SeasonalForecaster(lag_value=7)
        forecaster.fit(train_df)
        result = forecaster.predict(test_df)
        assert result[f"seasonal_forecast_lag_7"].is_null().sum() == 0

    def test_first_test_rows_use_training_tail(self):
        # Train ends at 100, 101, 102, 103, 104, 105, 106 (last 7 values)
        train_df = pl.DataFrame({"demand": [float(i) for i in range(100, 107)]})
        test_df = pl.DataFrame({"demand": [200.0, 201.0, 202.0]})
        forecaster = SeasonalForecaster(lag_value=7)
        forecaster.fit(train_df)
        result = forecaster.predict(test_df)
        forecasts = result["seasonal_forecast_lag_7"].to_list()
        # First 3 test rows: lag-7 pulls from training tail [100, 101, 102]
        assert forecasts[0] == 100.0
        assert forecasts[1] == 101.0
        assert forecasts[2] == 102.0

    def test_predict_before_fit_raises(self):
        forecaster = SeasonalForecaster(lag_value=7)
        with pytest.raises(RuntimeError, match="predict\\(\\) called before fit\\(\\)"):
            forecaster.predict(pl.DataFrame({"demand": [1.0, 2.0]}))
