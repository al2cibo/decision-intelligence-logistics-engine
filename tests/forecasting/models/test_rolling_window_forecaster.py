import pytest
import polars as pl
from forecasting.models.rolling_window_forecaster import RollingWindowForecaster


class TestRollingWindowForecaster:

    def test_rolling_window(self):
        train_data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        train_df = pl.DataFrame({"demand": train_data})
        test_df = pl.DataFrame({"demand": [80.0, 90.0]})

        forecaster = RollingWindowForecaster(rolling_window=7, target_col="demand")
        forecaster.fit(train_df)
        result = forecaster.predict(test_df)

        forecast = result["ma_7_forecast"].to_list()

        # Training tail fills the warm-up window — no nulls in the test output
        assert None not in forecast
        assert len(forecast) == 2

        # First test row: rolling mean of the last 7 training values
        expected_first = sum(train_data) / 7
        assert forecast[0] == pytest.approx(expected_first, rel=1e-6)

        # Second test row: rolling mean of [train[-6]..train[-1], test[0]]
        expected_second = sum([20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]) / 7
        assert forecast[1] == pytest.approx(expected_second, rel=1e-6)

    def test_predict_before_fit_raises(self):
        forecaster = RollingWindowForecaster(rolling_window=7)
        with pytest.raises(RuntimeError, match="predict\\(\\) called before fit\\(\\)"):
            forecaster.predict(pl.DataFrame({"demand": [1.0, 2.0, 3.0]}))

    def test_no_nulls_in_forecast(self):
        train_df = pl.DataFrame({"demand": [float(i) for i in range(1, 15)]})
        test_df = pl.DataFrame({"demand": [float(i) for i in range(15, 20)]})
        forecaster = RollingWindowForecaster(rolling_window=7)
        forecaster.fit(train_df)
        result = forecaster.predict(test_df)
        assert result["ma_7_forecast"].is_null().sum() == 0
