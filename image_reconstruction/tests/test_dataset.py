from unittest import TestCase
import numpy as np
from image_reconstruction.dataset import ImgDataset


class TestImgDataset(TestCase):
    def setUp(self) -> None:
        self.img = np.random.rand(16, 16, 3)
        self.dataset = ImgDataset(self.img)
        return super().setUp()

    def test_len(self) -> None:
        self.assertEqual(len(self.dataset), self.img.shape[0] * self.img.shape[1])

    def test_getitem(self) -> None:
        h_pos = self.img.shape[0] // 2
        self.assertTrue(np.isclose(self.dataset[h_pos][1], self.img[0, h_pos]).all())
