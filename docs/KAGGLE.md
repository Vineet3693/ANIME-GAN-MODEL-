# Kaggle Workflow

This project supports two Kaggle dataset flows.

## Flow 1: Kaggle Notebook Mounted Dataset

Use this when training directly inside a Kaggle notebook.

1. Create a Kaggle notebook.
2. Enable GPU in the notebook settings.
3. Add the dataset `splcher/animefacedataset` from the notebook sidebar.
4. Upload or clone this repository.
5. Install any missing dependencies:

```bash
pip install -r requirements.txt
```

6. Verify the mounted dataset path:

```bash
python gan_anime_faces.py download-data
```

7. Train:

```bash
python gan_anime_faces.py train \
  --data-dir /kaggle/input/animefacedataset \
  --output-dir outputs \
  --epochs 50 \
  --batch-size 128 \
  --save-every 5
```

## Flow 2: Kaggle API Download

Use this when you want a reproducible API-based data import.

### Credentials

Provide Kaggle credentials in one of these ways:

- environment variables from `configs/kaggle.env.example`
- `~/.kaggle/kaggle.json`

### Commands

```bash
pip install -r requirements.txt
python gan_anime_faces.py download-data \
  --target-dir data/animefacedataset \
  --dataset splcher/animefacedataset \
  --force-api
python gan_anime_faces.py train \
  --data-dir data/animefacedataset \
  --output-dir outputs \
  --epochs 50 \
  --batch-size 128 \
  --save-every 5
```

## Evaluation

If `clean-fid` needs to download Inception weights and Kaggle internet is disabled, train on Kaggle and run the final `evaluate` command later in a runtime with internet access.
