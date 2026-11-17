"""Make the repo's plugin module and vendored scripts importable to the tests."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(autouse=True)
def isolate_classifier_budget(monkeypatch, tmp_path, request):
    """Keep test accounting off the real host and preserve legacy classifier doubles."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    router = getattr(request.module, "pl", None)
    if router is None or not hasattr(router, "CLASSIFIER_BUDGET_PATH"):
        return
    monkeypatch.setattr(router, "CLASSIFIER_BUDGET_PATH", str(tmp_path / "classifier.sqlite3"))

    def use_legacy_classifier(session_id, features, cfg):
        pool, confidence = router._classify(features, cfg)
        telemetry = {"excerpt_sent": bool(features.get("task_excerpt")),
                     "classifier_cost_usd": None, "classifier_usage": None}
        return (pool, confidence, telemetry), None

    # Existing routing tests stub _classify, not the external client. Test the
    # actual budget wrapper separately in test_classifier_controls.py.
    monkeypatch.setattr(router, "_budget_call", use_legacy_classifier)
