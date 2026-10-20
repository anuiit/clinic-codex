from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from codex_pipeline.data.tlachia.core import artifact


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/materialize_tlachia_snapshot.py"
SPEC = importlib.util.spec_from_file_location("materialize_tlachia_snapshot", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_materializer_refuses_simulation_with_failed_gate(tmp_path: Path) -> None:
    simulation = artifact(
        "tlachia-standalone-snapshot-simulation.v1",
        {
            "gates": {
                "full_286_class_coverage": True,
                "minimum_train_support": True,
                "zero_unresolved_label_conflicts": False,
                "zero_unresolved_business_test_overlap": True,
                "quarantine_accounting_complete": True,
                "source_groups_split_disjoint": True,
                "all_splits_nonempty": True,
                "research_snapshot_materialization_ready": False,
            }
        },
    )
    media = artifact("tlachia-media-download-manifest.v1", {})
    simulation_path = tmp_path / "simulation.json"
    media_path = tmp_path / "media.json"
    import json

    simulation_path.write_text(json.dumps(simulation), encoding="utf-8")
    media_path.write_text(json.dumps(media), encoding="utf-8")
    with pytest.raises(MODULE.IntegrityError, match="not materialization-ready"):
        MODULE.build_manifest(
            simulation_path=simulation_path,
            media_manifest_path=media_path,
            runtime_config_path=tmp_path / "unused-config.json",
        )
