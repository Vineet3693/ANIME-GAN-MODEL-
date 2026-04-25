#!/usr/bin/env python3
"""Train, sample, and evaluate a DCGAN for anime face generation."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_TORCH_IMPORT_ERROR: Exception | None = None
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.utils import save_image
except ImportError as exc:  # pragma: no cover - dependency availability differs by runtime
    torch = None
    nn = None
    DataLoader = None
    Dataset = None
    transforms = None
    save_image = None
    _TORCH_IMPORT_ERROR = exc

_MATPLOTLIB_IMPORT_ERROR: Exception | None = None
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - dependency availability differs by runtime
    plt = None
    _MATPLOTLIB_IMPORT_ERROR = exc

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_KAGGLE_DATASET = "splcher/animefacedataset"


def current_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive number")
    return parsed


def dataset_slug_name(dataset_slug: str) -> str:
    return dataset_slug.rstrip("/").split("/")[-1]


def normalize_dataset_name(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def require_torch_runtime(command_name: str) -> None:
    if torch is None or nn is None or DataLoader is None or Dataset is None or transforms is None or save_image is None:
        detail = f"{_TORCH_IMPORT_ERROR}" if _TORCH_IMPORT_ERROR else "torch and torchvision are unavailable"
        raise SystemExit(
            f"{command_name} requires PyTorch and torchvision. "
            f"Install the packages from requirements.txt before running this command. "
            f"Original import error: {detail}"
        )


def require_cleanfid() -> Any:
    try:
        from cleanfid import fid
    except ImportError as exc:
        raise SystemExit(
            "evaluate requires clean-fid. Install the packages from requirements.txt before running evaluation."
        ) from exc
    return fid


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def list_image_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {data_dir}")

    image_paths = sorted(
        path for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No supported image files were found under {data_dir}. "
            f"Expected extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )
    return image_paths


def find_kaggle_mounted_dataset(dataset_slug: str, kaggle_input_root: Path = Path("/kaggle/input")) -> Path | None:
    if not kaggle_input_root.exists() or not kaggle_input_root.is_dir():
        return None

    slug_name = dataset_slug_name(dataset_slug)
    normalized_slug = normalize_dataset_name(slug_name)
    candidate_dirs: list[Path] = []

    direct_match = kaggle_input_root / slug_name
    if direct_match.exists() and direct_match.is_dir():
        candidate_dirs.append(direct_match)

    for child in sorted(kaggle_input_root.iterdir()):
        if not child.is_dir():
            continue
        normalized_child = normalize_dataset_name(child.name)
        if normalized_slug in normalized_child or normalized_child in normalized_slug:
            candidate_dirs.append(child)

    seen: set[Path] = set()
    for candidate in candidate_dirs:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            list_image_files(candidate)
            return candidate
        except FileNotFoundError:
            continue

    return None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_output_dirs(output_root: Path) -> dict[str, Path]:
    output_root = ensure_dir(output_root)
    checkpoints_dir = ensure_dir(output_root / "checkpoints")
    samples_dir = ensure_dir(output_root / "samples")
    reports_dir = ensure_dir(output_root / "reports")
    return {
        "root": output_root,
        "checkpoints": checkpoints_dir,
        "samples": samples_dir,
        "reports": reports_dir,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def save_loss_curve(history: list[dict[str, float]], output_path: Path) -> None:
    if plt is None:
        warning_path = output_path.with_suffix(".txt")
        warning_path.write_text(
            "matplotlib is not installed in this environment, so loss_curve.png was not generated.\n",
            encoding="utf-8",
        )
        return

    epochs = [entry["epoch"] for entry in history]
    d_losses = [entry["discriminator_loss"] for entry in history]
    g_losses = [entry["generator_loss"] for entry in history]

    figure = plt.figure(figsize=(8, 5))
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(epochs, d_losses, label="Discriminator Loss", linewidth=2)
    axis.plot(epochs, g_losses, label="Generator Loss", linewidth=2)
    axis.set_title("DCGAN Training Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def clean_directory(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


if torch is not None:

    class AnimeFaceDataset(Dataset):
        def __init__(self, data_dir: Path, image_size: int) -> None:
            self.image_paths = list_image_files(data_dir)
            self.transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

        def __len__(self) -> int:
            return len(self.image_paths)

        def __getitem__(self, index: int) -> Any:
            image_path = self.image_paths[index]
            with Image.open(image_path) as image:
                image = image.convert("RGB")
            return self.transform(image)


    class Generator(nn.Module):
        def __init__(self, latent_dim: int, base_channels: int = 64) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, base_channels * 8, kernel_size=4, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(base_channels * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(True),
                nn.ConvTranspose2d(base_channels, 3, kernel_size=4, stride=2, padding=1, bias=False),
                nn.Tanh(),
            )

        def forward(self, latent: Any) -> Any:
            return self.network(latent)


    class Discriminator(nn.Module):
        def __init__(self, base_channels: int = 64) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(3, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_channels * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_channels * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_channels * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            )

        def forward(self, images: Any) -> Any:
            return self.network(images).view(-1)


else:

    class AnimeFaceDataset:  # pragma: no cover - used only when dependencies are missing
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_torch_runtime("dataset loading")


    class Generator:  # pragma: no cover - used only when dependencies are missing
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_torch_runtime("generator initialization")


    class Discriminator:  # pragma: no cover - used only when dependencies are missing
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_torch_runtime("discriminator initialization")


def weights_init(module: Any) -> None:
    class_name = module.__class__.__name__
    if "Conv" in class_name:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in class_name:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


def select_device() -> Any:
    require_torch_runtime("device selection")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_training_grid(generator: Any, fixed_noise: Any, output_path: Path) -> None:
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise)
        save_image(fake_images, output_path, nrow=8, normalize=True, value_range=(-1, 1))
    generator.train()


def build_checkpoint_payload(
    epoch: int,
    generator: Any,
    discriminator: Any,
    optimizer_g: Any,
    optimizer_d: Any,
    history: list[dict[str, float]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "timestamp": current_timestamp(),
        "config": {
            "image_size": args.image_size,
            "latent_dim": args.latent_dim,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "save_every": args.save_every,
            "seed": args.seed,
            "num_workers": args.num_workers,
            "generator_base_channels": 64,
            "discriminator_base_channels": 64,
        },
        "history": history,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "optimizer_d_state_dict": optimizer_d.state_dict(),
    }


def load_checkpoint(checkpoint_path: Path, device: Any) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)


def build_generator_from_checkpoint(checkpoint: dict[str, Any], device: Any) -> tuple[Any, dict[str, Any]]:
    config = checkpoint.get("config", {})
    latent_dim = int(config.get("latent_dim", 100))
    generator = Generator(latent_dim=latent_dim).to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    return generator, config


def save_generated_batch(fake_images: Any, target_dir: Path, start_index: int) -> int:
    for offset, image_tensor in enumerate(fake_images):
        output_path = target_dir / f"sample_{start_index + offset:05d}.png"
        save_image(image_tensor, output_path, normalize=True, value_range=(-1, 1))
    return start_index + len(fake_images)


def command_train(args: argparse.Namespace) -> None:
    require_torch_runtime("train")
    set_seed(args.seed)

    data_dir = args.data_dir.expanduser().resolve()
    output_dirs = prepare_output_dirs(args.output_dir.expanduser())
    device = select_device()

    dataset = AnimeFaceDataset(data_dir=data_dir, image_size=args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    generator = Generator(latent_dim=args.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=device)

    history: list[dict[str, float]] = []
    metrics_path = output_dirs["reports"] / "training_metrics.json"
    latest_checkpoint_path = output_dirs["checkpoints"] / "latest.pt"

    print(f"Training on device: {device}")
    print(f"Found {len(dataset)} training images under {data_dir}")

    for epoch in range(1, args.epochs + 1):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        batch_count = 0

        for real_images in dataloader:
            real_images = real_images.to(device)
            current_batch_size = real_images.size(0)
            real_labels = torch.ones(current_batch_size, device=device)
            fake_labels = torch.zeros(current_batch_size, device=device)

            optimizer_d.zero_grad(set_to_none=True)
            real_logits = discriminator(real_images)
            d_loss_real = criterion(real_logits, real_labels)

            noise = torch.randn(current_batch_size, args.latent_dim, 1, 1, device=device)
            fake_images = generator(noise)
            fake_logits = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_logits, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            fooled_logits = discriminator(fake_images)
            g_loss = criterion(fooled_logits, real_labels)
            g_loss.backward()
            optimizer_g.step()

            epoch_d_loss += float(d_loss.item())
            epoch_g_loss += float(g_loss.item())
            batch_count += 1

        avg_d_loss = epoch_d_loss / max(batch_count, 1)
        avg_g_loss = epoch_g_loss / max(batch_count, 1)
        history.append(
            {
                "epoch": float(epoch),
                "discriminator_loss": avg_d_loss,
                "generator_loss": avg_g_loss,
            }
        )

        sample_grid_path = output_dirs["samples"] / f"epoch_{epoch:03d}_grid.png"
        save_training_grid(generator, fixed_noise, sample_grid_path)

        checkpoint_payload = build_checkpoint_payload(
            epoch=epoch,
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            history=history,
            args=args,
        )
        torch.save(checkpoint_payload, latest_checkpoint_path)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            milestone_path = output_dirs["checkpoints"] / f"gan_epoch_{epoch:03d}.pt"
            torch.save(checkpoint_payload, milestone_path)

        write_json(
            metrics_path,
            {
                "command": "train",
                "timestamp": current_timestamp(),
                "data_dir": str(data_dir),
                "output_dir": str(output_dirs["root"]),
                "device": str(device),
                "dataset_size": len(dataset),
                "epochs_completed": epoch,
                "history": history,
            },
        )

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f} | "
            f"Sample Grid: {sample_grid_path.name}"
        )

    loss_curve_path = output_dirs["reports"] / "loss_curve.png"
    save_loss_curve(history, loss_curve_path)
    print(f"Training finished. Latest checkpoint: {latest_checkpoint_path}")


def command_sample(args: argparse.Namespace) -> None:
    require_torch_runtime("sample")

    output_dirs = prepare_output_dirs(args.output_dir.expanduser())
    device = select_device()
    checkpoint = load_checkpoint(args.checkpoint.expanduser(), device)
    generator, config = build_generator_from_checkpoint(checkpoint, device)
    latent_dim = int(config.get("latent_dim", 100))

    generated_dir = clean_directory(output_dirs["samples"] / "generated_samples")
    total_saved = 0
    batch_size = min(64, args.num_samples)

    with torch.no_grad():
        while total_saved < args.num_samples:
            current_batch_size = min(batch_size, args.num_samples - total_saved)
            noise = torch.randn(current_batch_size, latent_dim, 1, 1, device=device)
            fake_images = generator(noise)
            total_saved = save_generated_batch(fake_images, generated_dir, total_saved)

    grid_count = min(args.num_samples, 64)
    with torch.no_grad():
        noise = torch.randn(grid_count, latent_dim, 1, 1, device=device)
        fake_images = generator(noise)
        nrow = max(1, int(math.sqrt(grid_count)))
        save_image(
            fake_images,
            output_dirs["samples"] / "generated_grid.png",
            nrow=nrow,
            normalize=True,
            value_range=(-1, 1),
        )

    write_json(
        output_dirs["reports"] / "sample_metadata.json",
        {
            "command": "sample",
            "timestamp": current_timestamp(),
            "checkpoint": str(args.checkpoint.expanduser()),
            "output_dir": str(output_dirs["root"]),
            "num_samples": args.num_samples,
            "device": str(device),
            "latent_dim": latent_dim,
            "samples_directory": str(generated_dir),
        },
    )
    print(f"Generated {args.num_samples} samples into {generated_dir}")


def command_evaluate(args: argparse.Namespace) -> None:
    require_torch_runtime("evaluate")
    fid = require_cleanfid()

    data_dir = args.data_dir.expanduser().resolve()
    output_dirs = prepare_output_dirs(args.output_dir.expanduser())
    device = select_device()
    checkpoint = load_checkpoint(args.checkpoint.expanduser(), device)
    generator, config = build_generator_from_checkpoint(checkpoint, device)
    latent_dim = int(config.get("latent_dim", 100))

    all_real_images = list_image_files(data_dir)
    sample_count = min(args.eval_samples, len(all_real_images))
    if sample_count < 2:
        raise SystemExit("FID evaluation requires at least two real images.")

    real_dir = clean_directory(output_dirs["reports"] / "fid_real_subset")
    fake_dir = clean_directory(output_dirs["reports"] / "fid_fake_subset")
    rng = random.Random(42)
    selected_real_images = (
        rng.sample(all_real_images, sample_count) if len(all_real_images) > sample_count else all_real_images[:sample_count]
    )

    for index, image_path in enumerate(selected_real_images):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.save(real_dir / f"real_{index:05d}.png")

    total_saved = 0
    batch_size = min(64, sample_count)
    with torch.no_grad():
        while total_saved < sample_count:
            current_batch_size = min(batch_size, sample_count - total_saved)
            noise = torch.randn(current_batch_size, latent_dim, 1, 1, device=device)
            fake_images = generator(noise)
            for offset, image_tensor in enumerate(fake_images):
                save_image(
                    image_tensor,
                    fake_dir / f"fake_{total_saved + offset:05d}.png",
                    normalize=True,
                    value_range=(-1, 1),
                )
            total_saved += current_batch_size

    fid_score = float(fid.compute_fid(str(real_dir), str(fake_dir), mode="clean"))
    metrics_payload = {
        "command": "evaluate",
        "timestamp": current_timestamp(),
        "checkpoint": str(args.checkpoint.expanduser()),
        "data_dir": str(data_dir),
        "output_dir": str(output_dirs["root"]),
        "device": str(device),
        "eval_samples_requested": args.eval_samples,
        "eval_samples_used": sample_count,
        "fid_score": fid_score,
        "real_subset_dir": str(real_dir),
        "fake_subset_dir": str(fake_dir),
    }
    write_json(output_dirs["reports"] / "evaluation_metrics.json", metrics_payload)
    print(f"FID score: {fid_score:.4f}")


def command_download_data(args: argparse.Namespace) -> None:
    dataset_slug = args.dataset
    target_dir = ensure_dir(args.target_dir.expanduser())
    kaggle_input_root = args.kaggle_input_root.expanduser()

    mounted_dataset = None if args.force_api else find_kaggle_mounted_dataset(dataset_slug, kaggle_input_root)
    if mounted_dataset is not None:
        write_json(
            target_dir / "download_metadata.json",
            {
                "command": "download-data",
                "timestamp": current_timestamp(),
                "source": "mounted_input",
                "dataset_slug": dataset_slug,
                "kaggle_input_root": str(kaggle_input_root),
                "requested_target_dir": str(target_dir),
                "resolved_data_dir": str(mounted_dataset),
            },
        )
        print(f"Kaggle mounted dataset detected at {mounted_dataset}")
        print(f"Use this path for training: {mounted_dataset}")
        return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit(
            "download-data requires the kaggle package. Install the packages from requirements.txt "
            "or attach the dataset through the Kaggle notebook sidebar."
        ) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        raise SystemExit(
            "Kaggle API authentication failed. Set KAGGLE_USERNAME and KAGGLE_KEY or place kaggle.json "
            "in ~/.kaggle/ before running download-data."
        ) from exc

    api.dataset_download_files(dataset_slug, path=str(target_dir), unzip=True, quiet=False)

    try:
        discovered_images = len(list_image_files(target_dir))
    except FileNotFoundError:
        discovered_images = 0

    write_json(
        target_dir / "download_metadata.json",
        {
            "command": "download-data",
            "timestamp": current_timestamp(),
            "source": "kaggle_api",
            "dataset_slug": dataset_slug,
            "requested_target_dir": str(target_dir),
            "resolved_data_dir": str(target_dir),
            "discovered_image_count": discovered_images,
        },
    )
    print(f"Downloaded dataset {dataset_slug} into {target_dir}")
    print(f"Use this path for training: {target_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anime face generation with a DCGAN baseline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the DCGAN on an anime face dataset.")
    train_parser.add_argument("--data-dir", type=Path, required=True, help="Path to the extracted anime face dataset.")
    train_parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for checkpoints and reports.")
    train_parser.add_argument("--epochs", type=positive_int, default=50, help="Number of training epochs.")
    train_parser.add_argument("--batch-size", type=positive_int, default=128, help="Training batch size.")
    train_parser.add_argument(
        "--image-size",
        type=positive_int,
        default=64,
        choices=[64],
        help="Training image size. This baseline is configured for 64x64 images.",
    )
    train_parser.add_argument("--latent-dim", type=positive_int, default=100, help="Latent vector size.")
    train_parser.add_argument("--lr", type=positive_float, default=0.0002, help="Learning rate for both optimizers.")
    train_parser.add_argument("--num-workers", type=int, default=2, help="Number of DataLoader worker processes.")
    train_parser.add_argument("--save-every", type=positive_int, default=5, help="Checkpoint save interval in epochs.")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    train_parser.set_defaults(func=command_train)

    sample_parser = subparsers.add_parser("sample", help="Generate images from a trained checkpoint.")
    sample_parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a saved checkpoint.")
    sample_parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for generated samples.")
    sample_parser.add_argument("--num-samples", type=positive_int, default=64, help="Number of images to generate.")
    sample_parser.set_defaults(func=command_sample)

    evaluate_parser = subparsers.add_parser("evaluate", help="Compute FID against a real image subset.")
    evaluate_parser.add_argument("--data-dir", type=Path, required=True, help="Path to the extracted anime face dataset.")
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a saved checkpoint.")
    evaluate_parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for evaluation artifacts.")
    evaluate_parser.add_argument("--eval-samples", type=positive_int, default=5000, help="Number of real and fake images for FID.")
    evaluate_parser.set_defaults(func=command_evaluate)

    download_parser = subparsers.add_parser(
        "download-data",
        help="Resolve or download the Kaggle anime face dataset for training.",
    )
    download_parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("data") / dataset_slug_name(DEFAULT_KAGGLE_DATASET),
        help="Directory where Kaggle API downloads should be extracted and where metadata will be written.",
    )
    download_parser.add_argument(
        "--dataset",
        default=DEFAULT_KAGGLE_DATASET,
        help="Kaggle dataset slug to use for download or mounted dataset detection.",
    )
    download_parser.add_argument(
        "--kaggle-input-root",
        type=Path,
        default=Path("/kaggle/input"),
        help="Root directory used to detect datasets already attached to a Kaggle notebook.",
    )
    download_parser.add_argument(
        "--force-api",
        action="store_true",
        help="Skip mounted dataset detection and force a Kaggle API download.",
    )
    download_parser.set_defaults(func=command_download_data)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Interrupted by user.") from None
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc
