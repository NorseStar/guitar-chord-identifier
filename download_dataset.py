"""Download the supplied guitar-chord dataset into data/raw."""

from pathlib import Path
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "data" / "raw"

def main():
    """Download the dataset files, reusing any files already cached."""
    snapshot_download(
        repo_id="severyn-k/isolated-guitar-chords",
        repo_type="dataset",
        local_dir=str(DATASET_DIR)
    )

    print(f"Dataset downloaded to {DATASET_DIR}")

if __name__ == "__main__":
    main()