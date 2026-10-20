"""Preview or explicitly import legacy review-index.json into SQLite."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.services.annotation_review import AnnotationReviewStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Import after creating and verifying a backup")
    args = parser.parse_args()
    store = AnnotationReviewStore(args.annotations_dir)
    result = store.import_legacy_manifest() if args.apply else store.preview_legacy_import()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
