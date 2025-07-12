import os
import sys
import zipfile
import tempfile
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.visualize_salient_thoughts import visualize_salient_thoughts


def process_zip(zip_path: Path, output_root: Path):
    """Extract a single zip file, generate visualization, and clean up."""
    name_parts = zip_path.stem.split("_")
    if len(name_parts) < 4:
        print(f"❌ Unexpected filename format: {zip_path.name}")
        return

    # Example filename pattern: Qwen3_aime2024_0_0-shot-new.zip
    _, dataset_token, example_idx, *_ = name_parts
    dataset = "aime" if "aime" in dataset_token else "math-algebra"
    example_name = f"{dataset}_{example_idx}"

    # Each example gets its own sub-folder for unique output filenames
    example_output_dir = output_root / example_name
    example_output_dir.mkdir(parents=True, exist_ok=True)

    # Skip if this example was already processed
    if (example_output_dir / "salient_thoughts.png").exists():
        print(f"↪️  {example_name} already processed. Skipping…")
        return

    # Safely extract to a temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)

        # Locate salient_thoughts.npy inside the extracted contents
        extracted_dir = Path(tmp_dir)
        npy_candidates = list(extracted_dir.rglob("salient_thoughts.npy"))
        if not npy_candidates:
            print(f"❌ salient_thoughts.npy not found in {zip_path.name}")
            return

        npy_path = npy_candidates[0]
        try:
            data = np.load(npy_path)
        except Exception as e:
            print(f"❌ Failed loading {npy_path}: {e}")
            return

        # Generate and save visualization
        print(f"✅ Processing {example_name} ({dataset}) …")
        visualize_salient_thoughts(
            data=data,
            title=example_name,
            output_dir=str(example_output_dir),
        )


def batch_process_all(base_dir: Path, output_root: Path):
    """Iterate over every *.zip file in base_dir and process sequentially."""
    zip_files = sorted(base_dir.glob("*.zip"))
    if not zip_files:
        print(f"No zip files found in {base_dir}")
        return

    print(f"🔍 Found {len(zip_files)} zip files. Starting batch processing…")
    for i, zip_path in enumerate(zip_files, 1):
        print(f"\n[{i:02}/{len(zip_files)}] {zip_path.name}")
        process_zip(zip_path, output_root)


if __name__ == "__main__":
    BASE_DIR = Path("../output4_new")
    OUTPUT_ROOT = Path("../visualiztions_salient_thoughts/qwen3/thought_level_analysis")
    batch_process_all(BASE_DIR, OUTPUT_ROOT) 