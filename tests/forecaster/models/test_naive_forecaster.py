import polars
import pytest

from forecasting.models.naive_forecaster import NaiveForecaster


class TestNaiveForecaster:

    def test_valid_df_naive_forecaster(self):
        train_df = polars.DataFrame({"demand": [10.2, 13, 14.5]})
        test_df = polars.DataFrame({"demand": [10.2, 13, 14.5]})

        forecaster = NaiveForecaster(target_col="demand")
        forecaster.fit(train_df)
        forecast_df = forecaster.predict(test_df)

        assert forecast_df.shape == (3, 2)
        # First test row gets the last training value (14.5)
        assert forecast_df[0, "naive_forecast"] == 14.5
        # Second test row gets the first test actual (10.2)
        assert forecast_df[1, "naive_forecast"] == 10.2

    def test_predict_before_fit_raises(self):
        forecaster = NaiveForecaster()
        with pytest.raises(RuntimeError, match="predict\\(\\) called before fit\\(\\)"):
            forecaster.predict(polars.DataFrame({"demand": [1.0, 2.0]}))

    def test_no_nulls_in_forecast(self):
        train_df = polars.DataFrame({"demand": [5.0, 6.0, 7.0]})
        test_df = polars.DataFrame({"demand": [8.0, 9.0, 10.0]})
        forecaster = NaiveForecaster()
        forecaster.fit(train_df)
        result = forecaster.predict(test_df)
        assert result["naive_forecast"].is_null().sum() == 0
