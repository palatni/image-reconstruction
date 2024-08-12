from unittest import TestCase
import numpy as np
from image_reconstruction.accumulators import MSEMetricsAccumulator, ImgDataAccumulator



class TestMSEMetricsAccumulator(TestCase):
    def setUp(self) -> None:
        self._accumulator = MSEMetricsAccumulator()
        return super().setUp()

    def test_mse(self) -> None:
        self._accumulator.reset()
        mock_mse = 0.1
        mock_samples = 4
        for _ in range(10):
            self._accumulator.update(mock_mse, mock_samples)
        self.assertAlmostEqual(self._accumulator.mse, mock_mse, delta=1e-7)

    def test_psnr(self) -> None:
        self._accumulator.reset()
        mock_mse = 0.1
        mock_samples = 4
        mock_psnr = 10
        for _ in range(10):
            self._accumulator.update(mock_mse, mock_samples)
        self.assertAlmostEqual(self._accumulator.psnr, mock_psnr, delta=1e-7)


class TestImgDataAccumulator(TestCase):
    def setUp(self) -> None:
        self._test_shape = (16, 16, 3)
        self._accumulator = ImgDataAccumulator(self._test_shape)
        return super().setUp()

    def test_reset(self) -> None:
        self._accumulator.reset()
        self.assertTrue(not self._accumulator.img.any())
        self._accumulator.reset()

    def test_random_img(self) -> None:
        self._accumulator.reset()
        img = np.random.rand(*self._test_shape)

        pos_1_mask = np.random.rand(*img.shape[:-1]) > 0.5
        pos_2_mask = np.logical_not(pos_1_mask)
        for mask in (pos_1_mask, pos_2_mask):
            pos = np.vstack(np.where(mask), dtype=np.float32).T
            pos[:, 0] /= img.shape[0]
            pos[:, 1] /= img.shape[1]
            channel_data = img[mask]
            self._accumulator.update(pos, channel_data)

        self.assertTrue((self._accumulator.img == img).all())
