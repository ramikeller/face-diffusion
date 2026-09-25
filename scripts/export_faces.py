"""One-off dataset prep: not part of the Rust pipeline.

Downloads the landmark-aligned CelebA mirror hosted at Yuehao/celeba on
Hugging Face (the actual img_align_celeba.zip - eyes/nose/mouth at
consistent pixel positions across the dataset, not the unaligned
in-the-wild version) and extracts the first N images as plain JPEGs into
data/faces/, which src/data.rs's FaceDataset::load then reads directly.

Usage (from the project root, in a throwaway venv):
    python3 -m venv .venv && source .venv/bin/activate
    pip install huggingface_hub
    python3 scripts/export_faces.py
"""

import zipfile

from huggingface_hub import hf_hub_download

OUT_DIR = "data/faces"
N_IMAGES = 20_000

zip_path = hf_hub_download(
    repo_id="Yuehao/celeba",
    repo_type="dataset",
    filename="img_align_celeba.zip",
)
print(f"downloaded to {zip_path}")

with zipfile.ZipFile(zip_path) as zf:
    names = sorted(n for n in zf.namelist() if n.endswith(".jpg"))
    print(f"{len(names)} images in archive")
    subset = names[:N_IMAGES]
    for i, name in enumerate(subset):
        data = zf.read(name)
        out_name = f"{i:06d}.jpg"
        with open(f"{OUT_DIR}/{out_name}", "wb") as f:
            f.write(data)
        if (i + 1) % 2000 == 0:
            print(f"extracted {i + 1}/{N_IMAGES}")

print(f"done: {len(subset)} images written to {OUT_DIR}")
