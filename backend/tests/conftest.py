"""Shared test-process defaults.

Production keeps authentication enabled by default. Legacy and route
characterization tests run without user/session fixtures, so they opt out
explicitly for the test process and its subprocesses. Security tests override
or remove this variable when exercising the production default.
"""

from __future__ import annotations

import os

import pytest


os.environ.setdefault("AUTH_REQUIRED", "false")

RESEARCH_ARTIFACT_SKIP_REASON = "requires unshipped research artifact"


def pytest_configure(config: pytest.Config) -> None:
    # Also registered in backend/pytest.ini; repeated here so the repo-root CI
    # invocation (``pytest backend/tests scripts``) knows the marker too.
    config.addinivalue_line(
        "markers",
        "research_artifacts: needs untracked local research artifacts "
        "(checkpoints, reports/, .omc runs, training_runs, /mnt/f data); "
        "skipped when they are absent",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Tag every artifact-gated skipif with the ``research_artifacts`` marker.

    Lets ``-m "not research_artifacts"`` deselect them and keeps older tests
    that only carry the skipif in the same category.
    """
    for item in items:
        for mark in item.iter_markers("skipif"):
            if str(mark.kwargs.get("reason", "")).startswith(RESEARCH_ARTIFACT_SKIP_REASON):
                item.add_marker(pytest.mark.research_artifacts)
                break
