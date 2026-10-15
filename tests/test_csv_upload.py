"""
test_csv_upload.py
==================
Unit tests for the CSV telemetry parsing and real-time risk diagnostic endpoints.
"""

import io
import json
import numpy as np
import pytest
from unittest.mock import MagicMock
from web.app import app, _parse_csv_signal, _state


@pytest.fixture
def client():
    app.config["TESTING"] = True
    _state["ready"] = True
    mock_model = MagicMock()
    mock_model.score.return_value = np.array([0.25], dtype=np.float32)
    mock_model.confidence.return_value = 0.95
    mock_model.severity.return_value = "normal"
    _state["model"] = mock_model
    _state["feat_names"] = ["rms", "kurtosis"]
    _state["feat_stats"] = {"rms": (0.05, 0.01), "kurtosis": (3.0, 0.5)}
    with app.test_client() as client:
        yield client


def test_parse_csv_signal_single_column():
    csv_text = "0.012\n-0.034\n0.056\n-0.011\n"
    arr, load = _parse_csv_signal(csv_text)
    assert len(arr) == 4
    assert np.isclose(arr[0], 0.012)
    assert load is None


def test_parse_csv_signal_with_header_and_load():
    csv_text = (
        "time,vibration,load_hp\n"
        "0.000,0.045,2\n"
        "0.001,0.052,2\n"
        "0.002,-0.031,2\n"
    )
    arr, load = _parse_csv_signal(csv_text)
    assert len(arr) == 3
    assert np.isclose(arr[1], 0.052)
    assert load == 2


def test_sample_csv_endpoint(client):
    resp = client.get("/api/sample_csv?type=inner_race&load_hp=1")
    assert resp.status_code == 200
    assert "text/csv" in resp.content_type
    assert "attachment" in resp.headers.get("Content-Disposition", "")
    lines = resp.data.decode("utf-8").splitlines()
    assert len(lines) > 500
    assert "timestamp_sec,vibration_g,load_hp" in lines[0]


def test_upload_csv_empty_payload(client):
    resp = client.post("/api/upload_csv", json={"csv_text": ""})
    assert resp.status_code == 400
    data = json.loads(resp.data)
    assert "error" in data


def test_upload_csv_too_short(client):
    resp = client.post("/api/upload_csv", json={"csv_text": "0.1\n0.2\n0.3"})
    assert resp.status_code == 400
    data = json.loads(resp.data)
    assert "too few numeric data points" in data.get("error", "")
