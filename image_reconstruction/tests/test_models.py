from unittest import TestCase
from operator import mul
import torch
from image_reconstruction.models import FurierEncoder, FurierMLP


class TestFurierEncoder(TestCase):
    def test_output_shape(self) -> None:
        img_spatial_shape = (4, 4)
        num_phases = 10
        encoder = FurierEncoder(
            num_phases=num_phases,
            init_scale=1,
            trainable=False,
        )
        mock_input = torch.argwhere(torch.ones(img_spatial_shape)).float()
        mock_input /= torch.tensor(img_spatial_shape)[None, ...].float()
        output_shape = (mock_input.shape[0], num_phases * 2)
        encoder_output_shape = tuple(encoder(mock_input).shape)
        self.assertEqual(output_shape, encoder_output_shape)


class TestFurierMLP(TestCase):
    def test_output_shape(self) -> None:
        img_spatial_shape = (4, 4)
        num_phases = 10
        encoder = FurierMLP(
            num_phases=num_phases,
            encoder_scale=1,
            mlp_feature_list=[4, 4, 4],
            trainable_encoder=False,
        )
        mock_input = torch.argwhere(torch.ones(img_spatial_shape)).float()
        mock_input /= torch.tensor(img_spatial_shape)[None, ...].float()
        output_shape = (mock_input.shape[0], 3)
        encoder_output_shape = tuple(encoder(mock_input).shape)
        self.assertEqual(output_shape, encoder_output_shape)
