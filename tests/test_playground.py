"""Run with: python -m unittest discover -s tests -v"""
import csv
import io
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
import playground


def image_bytes(image, format='PNG'):
    out = io.BytesIO()
    image.save(out, format=format)
    return out.getvalue()


class UploadValidationTests(unittest.TestCase):
    def test_invalid_and_oversized_files(self):
        for body in (b'', b'not an image', b'x' * (playground.MAX_BYTES + 1)):
            with self.subTest(size=len(body)), self.assertRaises(ValueError):
                playground.decode_image(body)

    def test_supported_formats_and_grayscale(self):
        for format in ('PNG', 'JPEG', 'WEBP'):
            with self.subTest(format=format):
                decoded = playground.decode_image(image_bytes(Image.new('L', (13, 9), 100), format))
                self.assertEqual(decoded.mode, 'RGB')
                self.assertEqual(decoded.size, (13, 9))

    def test_transparent_pixels_are_composited_on_white(self):
        decoded = playground.decode_image(image_bytes(Image.new('RGBA', (2, 2), (0, 0, 0, 0))))
        self.assertEqual(decoded.getpixel((0, 0)), (255, 255, 255))

    def test_unsupported_format(self):
        with self.assertRaises(ValueError):
            playground.decode_image(image_bytes(Image.new('RGB', (2, 2)), 'GIF'))

    def test_pixel_limit(self):
        from unittest.mock import patch
        with patch.object(playground, 'MAX_PIXELS', 3), self.assertRaises(ValueError):
            playground.decode_image(image_bytes(Image.new('RGB', (2, 2))))


@unittest.skipUnless(playground.status()['ready'], 'Run app/prepare_playground.py for real-model checks')
class RealModelTests(unittest.TestCase):
    def test_larger_uploads_do_not_collapse_into_one_cluster(self):
        images = np.load(ROOT / 'image_data.npy', mmap_mode='r')
        with (ROOT / 'viz_data.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        examples = {}
        for row in rows:
            examples.setdefault(int(row['kmeans_cluster']), int(row['image_id']))
        predicted = set()
        for cluster, index in examples.items():
            original = Image.fromarray(images[index].astype('uint8'))
            for size in ((224, 224), (256, 256), (512, 256)):
                with self.subTest(cluster=cluster, size=size):
                    enlarged = original.resize(size, Image.Resampling.NEAREST)
                    result = playground.predict(image_bytes(enlarged))
                    self.assertEqual(result['cluster'], cluster)
                    self.assertEqual(result['neighbors'][0]['image_id'], index)
                    self.assertGreater(result['neighbors'][0]['score'], .999)
                    predicted.add(result['cluster'])
        self.assertEqual(predicted, set(examples))

    def test_dataset_images_retain_clusters_and_find_themselves(self):
        images = np.load(ROOT / 'image_data.npy', mmap_mode='r')
        with (ROOT / 'viz_data.csv').open(newline='') as stream:
            labels = {int(r['image_id']): int(r['kmeans_cluster']) for r in csv.DictReader(stream)}
        for index in (0, 500, 999):
            with self.subTest(image=index):
                result = playground.predict(image_bytes(Image.fromarray(images[index].astype('uint8'))))
                self.assertEqual(result['cluster'], labels[index])
                self.assertEqual(result['neighbors'][0]['image_id'], index)
                self.assertGreater(result['neighbors'][0]['score'], .999)
                self.assertEqual(len(result['neighbors']), 6)
                self.assertEqual(len({m['image_id'] for m in result['neighbors']}), 6)


if __name__ == '__main__':
    unittest.main()
