"""Rebuild missing derived PNG crops from immutable source images and boxes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.services.annotation_storage import clamp_bbox, normalize_bbox_to_int_pixels  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.annotations_dir.resolve()
    missing: list[tuple[Path, Path, tuple[int, int, int, int]]] = []
    for metadata_path in sorted(root.glob("*/metadata.json")):
        analysis_dir = metadata_path.parent
        if analysis_dir.is_symlink():
            raise ValueError(f"symlinked analysis directory: {analysis_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        image_path = analysis_dir / "image.png"
        with Image.open(image_path) as image:
            size = image.size
        for annotation in metadata["annotations"]:
            index = annotation["index"]
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError(f"invalid annotation index: {analysis_dir}")
            crop_path = analysis_dir / "elements" / f"{index}.png"
            if crop_path.exists():
                continue
            bbox = normalize_bbox_to_int_pixels(annotation["bbox"])
            if bbox != clamp_bbox(bbox, *size):
                raise ValueError(f"bbox exceeds source image: {analysis_dir.name}:{index}")
            missing.append((image_path, crop_path, bbox))
    if args.apply:
        for image_path, crop_path, (x, y, w, h) in missing:
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(image_path) as image, crop_path.open("xb") as output:
                image.crop((x, y, x + w, y + h)).save(output, format="PNG")
    print(json.dumps({"missing_crops": len(missing), "repaired": len(missing) if args.apply else 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
