#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

# Add backend/ to sys.path so services.annotation_storage is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from services.annotation_storage import save_annotation, decode_image_data_url


def main():
    parser = argparse.ArgumentParser(
        description="Export localStorage annotations to Elements/ training dataset"
    )
    parser.add_argument("input_json", help="Path to localStorage export JSON file")
    parser.add_argument(
        "--output",
        default="backend/training_data/Elements",
        help="Elements output directory (default: backend/training_data/Elements)",
    )
    parser.add_argument(
        "--annotations-dir",
        default="backend/annotations",
        help="Annotations base directory (default: backend/annotations)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_json)
    elements_dir = Path(args.output)
    annotations_dir = Path(args.annotations_dir)

    records = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        print("ERROR: input JSON must be a list of analysis records", file=sys.stderr)
        sys.exit(1)

    total_elements = 0
    total_classes: set[str] = set()
    errors = 0

    for i, record in enumerate(records, 1):
        analysis_id = record.get("id", "")
        user_annotations = record.get("annotations", {})

        if not user_annotations:
            print(f"[{i}/{len(records)}] Skipping {analysis_id} (no annotations)", file=sys.stderr)
            continue

        result_elements = record.get("result", {}).get("elements", [])
        image_data_url = record.get("imageDataUrl", "")

        ann_list = []
        for idx_str, class_name in user_annotations.items():
            idx = int(idx_str)
            if idx >= len(result_elements):
                print(f"  WARNING: index {idx} out of range for {analysis_id}", file=sys.stderr)
                continue
            bbox = result_elements[idx]["bbox"]
            ann_list.append({"index": idx, "bbox": bbox, "class_name": class_name})

        if not ann_list:
            print(f"[{i}/{len(records)}] Skipping {analysis_id} (no valid annotations)", file=sys.stderr)
            continue

        try:
            image = decode_image_data_url(image_data_url)
            result = save_annotation(
                analysis_id=analysis_id,
                image=image,
                annotations=ann_list,
                base_dir=annotations_dir,
                elements_dir=elements_dir,
            )
            count = result["saved_count"]
            classes = result["classes"]
            total_elements += count
            total_classes.update(classes)
            print(f"[{i}/{len(records)}] processed analysis_id={analysis_id} ({count} elements, classes={classes})", file=sys.stderr)
        except Exception as e:
            print(f"[{i}/{len(records)}] ERROR processing {analysis_id}: {e}", file=sys.stderr)
            errors += 1

    print(f"\nExported {total_elements} elements across {len(total_classes)} classes", file=sys.stderr)
    if errors:
        print(f"Errors: {errors} records failed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
