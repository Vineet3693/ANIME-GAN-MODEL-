#!/usr/bin/env python3
"""Kaggle entrypoint for DCGAN training."""

from __future__ import annotations

from pathlib import Path

import gan_anime_faces


def main() -> None:
    output_dir = Path("/kaggle/working/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    gan_anime_faces.main(
        [
            "train",
            "--data-dir",
            "/kaggle/input/animefacedataset",
            "--output-dir",
            str(output_dir),
            "--epochs",
            "50",
            "--batch-size",
            "128",
            "--image-size",
            "64",
            "--latent-dim",
            "100",
            "--lr",
            "0.0002",
            "--num-workers",
            "2",
            "--save-every",
            "5",
            "--seed",
            "42",
        ]
    )


if __name__ == "__main__":
    main()
