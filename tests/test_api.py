"""
tests/test_api.py
Basic automated tests used by the GitHub Actions CI/CD pipeline.
Run locally with:  pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.api import app


def test_health_endpoint():
    """The /health route must return 200 and status ok - this is the
    smoke test the CI pipeline runs before allowing a deploy."""
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_predict_requires_file():
    """/predict with no file attached should fail gracefully, not crash."""
    client = app.test_client()
    response = client.post("/predict")
    assert response.status_code == 400
