"""Add reviewable examples from images already shipped with the repository."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.annotation_storage import save_annotation  # noqa: E402


EXAMPLES = {
    "atl": ("026r_a_07-2.jpg", "033_02_01-7.jpg"),
    "calli": ("032_05_005-6.jpg", "032_05_009-6.jpg"),
    "tochtli": ("032_05_008-6.jpg", "032_05_012-6.jpg"),
}


def seed(annotations_dir: Path) -> None:
    source_dir = ROOT / "backend" / "data" / "glyphs_sample"
    for class_name, filenames in EXAMPLES.items():
        for filename in filenames:
            source = source_dir / f"{class_name}-glyph" / filename
            analysis_id = f"demo-{class_name}-{source.stem}"
            with Image.open(source) as source_image:
                image = source_image.convert("RGB")
            save_annotation(
                analysis_id,
                image,
                [{
                    "index": 0,
                    "class_name": class_name,
                    "bbox": [0, 0, image.width, image.height],
                    "note": "Exemple du corpus local : vérifier la classe avant validation.",
                }],
                annotations_dir,
                annotations_dir / "elements",
                image_name=filename,
            )
            print(f"{analysis_id}: annotation présente ; décision de revue inchangée")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", type=Path, default=ROOT / "backend" / "annotations")
    parser.add_argument("--apply", action="store_true", help="écrire les six annotations d'exemple")
    args = parser.parse_args()
    if args.apply:
        seed(args.annotations_dir)
    else:
        print("Aucune écriture. Ajouter --apply pour créer jusqu'à 6 annotations sans les valider.")
