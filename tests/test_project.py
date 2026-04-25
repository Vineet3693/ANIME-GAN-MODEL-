from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import gan_anime_faces as project


FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures"


class ValidatorTests(unittest.TestCase):
    def test_positive_int_accepts_valid_value(self) -> None:
        self.assertEqual(project.positive_int("5"), 5)

    def test_positive_int_rejects_zero(self) -> None:
        with self.assertRaises(Exception):
            project.positive_int("0")

    def test_positive_float_accepts_valid_value(self) -> None:
        self.assertEqual(project.positive_float("0.25"), 0.25)


class FilesystemTests(unittest.TestCase):
    def test_list_image_files_finds_nested_images(self) -> None:
        dataset_root = FIXTURES_ROOT / "dataset_root"
        discovered = project.list_image_files(dataset_root)
        self.assertEqual(discovered, [dataset_root / "nested" / "sample.png"])

    def test_prepare_output_dirs_uses_expected_paths(self) -> None:
        output_root = Path("outputs")
        with patch.object(project, "ensure_dir", side_effect=lambda path: path) as ensure_dir:
            output_dirs = project.prepare_output_dirs(output_root)

        self.assertEqual(output_dirs["root"], Path("outputs"))
        self.assertEqual(output_dirs["checkpoints"], Path("outputs") / "checkpoints")
        self.assertEqual(output_dirs["samples"], Path("outputs") / "samples")
        self.assertEqual(output_dirs["reports"], Path("outputs") / "reports")
        self.assertEqual(ensure_dir.call_count, 4)


class KaggleTests(unittest.TestCase):
    def test_find_kaggle_mounted_dataset_returns_matching_directory(self) -> None:
        kaggle_root = FIXTURES_ROOT / "kaggle_input"
        resolved = project.find_kaggle_mounted_dataset("splcher/animefacedataset", kaggle_root)
        self.assertEqual(resolved, kaggle_root / "animefacedataset")

    def test_download_data_uses_mounted_dataset_without_kaggle_package(self) -> None:
        args = project.build_parser().parse_args(
            [
                "download-data",
                "--target-dir",
                "data/animefacedataset",
                "--kaggle-input-root",
                str(FIXTURES_ROOT / "kaggle_input"),
            ]
        )

        with patch.object(project, "write_json") as write_json:
            args.func(args)

        self.assertTrue(write_json.called)
        payload = write_json.call_args.args[1]
        self.assertEqual(payload["source"], "mounted_input")
        self.assertIn("animefacedataset", payload["resolved_data_dir"])


class DependencyGuardTests(unittest.TestCase):
    def test_train_fails_cleanly_without_torch(self) -> None:
        if project.torch is not None:
            self.skipTest("Torch is installed in this environment; dependency guard test is not applicable.")

        with self.assertRaises(SystemExit) as context:
            project.main(["train", "--data-dir", str(FIXTURES_ROOT / "dataset_root")])

        self.assertIn("requires PyTorch and torchvision", str(context.exception))


class MockedPathTests(unittest.TestCase):
    def test_list_image_files_handles_mocked_paths(self) -> None:
        fake_root = MagicMock()
        fake_root.exists.return_value = True
        fake_root.is_dir.return_value = True
        image_path = MagicMock()
        image_path.is_file.return_value = True
        image_path.suffix = ".jpg"
        fake_root.rglob.return_value = [image_path]

        discovered = project.list_image_files(fake_root)

        self.assertEqual(discovered, [image_path])


if __name__ == "__main__":
    unittest.main()
