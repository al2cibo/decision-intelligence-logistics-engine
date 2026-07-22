"""API tests for optional destination holding_cost support on /optimize."""

from fastapi.testclient import TestClient
import pytest

from api.app import app, _to_destinations_df
from api.models import DestinationRecord

client = TestClient(app)


def _optimize_payload(*, destinations: list[dict], demand: float = 10.0) -> dict:
    return {
        "demand_ts": [
            {
                "date": "2026-06-01",
                "destination_id": "D1",
                "demand": demand,
            },
            {
                "date": "2026-06-02",
                "destination_id": "D1",
                "demand": demand,
            },
        ],
        "origins": [{"origin_id": "O1", "daily_capacity": 100.0}],
        "lanes": [{"origin_id": "O1", "destination_id": "D1", "unit_cost": 1.0}],
        "destinations": destinations,
        "initial_inventory": {"D1": 20.0},
    }


class TestToDestinationsDf:
    def test_omits_holding_cost_column_when_all_none(self):
        df = _to_destinations_df(
            [
                DestinationRecord(destination_id="D1"),
                DestinationRecord(destination_id="D2", holding_cost=None),
            ]
        )
        assert df.columns == ["destination_id"]

    def test_includes_holding_cost_and_defaults_missing_to_zero(self):
        df = _to_destinations_df(
            [
                DestinationRecord(destination_id="D1", holding_cost=1.5),
                DestinationRecord(destination_id="D2"),
            ]
        )
        assert df.columns == ["destination_id", "holding_cost"]
        by_id = {row["destination_id"]: row["holding_cost"] for row in df.to_dicts()}
        assert by_id == {"D1": 1.5, "D2": 0.0}


class TestOptimizeHoldingCost:
    def test_optimize_without_holding_cost_keeps_zero_holding(self):
        response = client.post(
            "/optimize",
            json=_optimize_payload(destinations=[{"destination_id": "D1"}]),
        )
        assert response.status_code == 200
        body = response.json()
        assert "transportation_cost" in body
        assert "holding_cost" in body
        assert body["holding_cost"] == 0.0
        assert body["total_cost"] == body["transportation_cost"]

    def test_optimize_with_holding_cost_affects_objective(self):
        # Excess initial inventory of 20 with demand 10+10 leaves inventory
        # carry-over in period 1 that incurs holding cost.
        without = client.post(
            "/optimize",
            json=_optimize_payload(destinations=[{"destination_id": "D1"}]),
        ).json()
        with_holding = client.post(
            "/optimize",
            json=_optimize_payload(
                destinations=[{"destination_id": "D1", "holding_cost": 2.0}]
            ),
        ).json()

        assert without["holding_cost"] == 0.0
        assert with_holding["holding_cost"] > 0.0
        assert with_holding["total_cost"] == pytest.approx(
            with_holding["transportation_cost"] + with_holding["holding_cost"]
        )
        assert with_holding["total_cost"] >= without["total_cost"]

    def test_optimize_negative_holding_cost_returns_422(self):
        response = client.post(
            "/optimize",
            json=_optimize_payload(
                destinations=[{"destination_id": "D1", "holding_cost": -1.0}]
            ),
        )
        assert response.status_code == 422
        assert "Negative holding_cost" in response.json()["detail"]
