"""Focused Module 3 index, text-cleaning, and matching checks."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from src.module3 import retrieval
from src.module3.intents import is_similar_retrieval_request


class _TinyEncoder:
    def eval(self):
        return self

    def encode_image(self, batch):
        # Red-channel average gives the fixtures stable, distinct embeddings.
        red = batch[:, 0].mean(dim=(1, 2))
        return torch.stack((red, 1.0 - red), dim=1)


def _preprocess(image):
    array = np.asarray(image.resize((8, 8)), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class Module3Tests(unittest.TestCase):
    def test_chat_phrasings_route_to_similar_retrieval(self):
        for phrase in (
            "retrieve similar cases", "retrive similar cases", "similar cases",
            "similar case", "previous case", "previous cases", "find similar images",
        ):
            with self.subTest(phrase=phrase):
                self.assertTrue(is_similar_retrieval_request(phrase))
        self.assertFalse(is_similar_retrieval_request("summarize the clinical narrative"))

    def test_clean_text_removes_labels_but_preserves_findings(self):
        self.assertEqual(
            retrieval.clean_text("Findings: Mild opacity. Impression: No acute process."),
            "Mild opacity. No acute process.",
        )

    def test_build_and_search_keep_image_id_and_original_text_together(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent / ".runtime") as temp:
            root = Path(temp)
            image_dir = root / "images"
            image_dir.mkdir()
            csv_path = root / "master.csv"
            index_path = root / "index.npz"
            Image.new("RGB", (12, 12), (255, 0, 0)).save(image_dir / "red.jpg")
            Image.new("RGB", (12, 12), (0, 0, 255)).save(image_dir / "blue.jpg")
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_id", "text", "text_augment"])
                writer.writeheader()
                writer.writerow({"image_id": "red", "text": "Findings: red source text", "text_augment": "wrong augmented text"})
                writer.writerow({"image_id": "blue", "text": "Impression: blue source text", "text_augment": "wrong augmented text"})
                writer.writerow({"image_id": "missing", "text": "Findings: absent image"})
            with patch.dict("os.environ", {
                "CXR_IMAGE_DIR": str(image_dir),
                "CXR_CSV_PATH": str(csv_path),
                "CXR_M3_INDEX_PATH": str(index_path),
            }), patch.object(retrieval, "load_encoder", return_value=(_TinyEncoder(), _preprocess)):
                stats = retrieval.build_index()
                self.assertEqual(stats["indexed"], 2)
                self.assertEqual(stats["missing_images"], 1)
                retriever = retrieval.ImageRetriever()
                results = retriever.search(image_dir / "red.jpg", limit=5)
            self.assertEqual([row["image_id"] for row in results], ["red", "blue"])
            self.assertEqual(results[0]["text"], "red source text")
            self.assertNotIn("augmented", results[0]["text"])
            self.assertLessEqual(len(results), 5)


if __name__ == "__main__":
    unittest.main()
