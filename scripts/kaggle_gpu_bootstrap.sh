#!/bin/bash
set -euo pipefail

pip install -r requirements.txt
python gan_anime_faces.py download-data --target-dir data/animefacedataset --dataset splcher/animefacedataset
python gan_anime_faces.py train \
  --data-dir /kaggle/input/animefacedataset \
  --output-dir outputs \
  --epochs 50 \
  --batch-size 128 \
  --save-every 5
