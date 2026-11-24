"""CPU checks for the image-adaptive filters. No dataset and no GPU."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'notebooks' / '_support'))
from ia_yolo import (  # noqa: E402
    FILTER_PARAMETERS, PREDICTOR_PARAMETERS, ImageAdaptive, apply_filters, defog,
    gaussian_kernel,
)


class ImageAdaptiveFilters(unittest.TestCase):
    def test_predictor_matches_the_paper_parameter_count(self):
        model = ImageAdaptive()
        count = sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(count, PREDICTOR_PARAMETERS)
        self.assertEqual(model.predictor.output.out_features, FILTER_PARAMETERS)
        self.assertTrue(torch.count_nonzero(model.predictor.output.weight) == 0)

    def test_zero_defog_keeps_the_image(self):
        image = torch.rand(2, 3, 32, 32)
        recovered = defog(image, torch.zeros(2, 1))
        self.assertTrue(torch.allclose(recovered, image))

    def test_identity_parameters_leave_the_image_unchanged(self):
        image = torch.rand(1, 3, 40, 48)
        raw = torch.zeros(1, FILTER_PARAMETERS)
        raw[:, 0] = -80
        raw[:, 5] = -80
        raw[:, 14] = -80
        filtered = apply_filters(image, raw, gaussian_kernel())
        self.assertTrue(torch.allclose(filtered, image, atol=1e-5))

    def test_detection_loss_can_train_the_predictor(self):
        image = torch.rand(2, 3, 64, 64)
        model = ImageAdaptive()
        filtered = model(image)
        filtered.mean().backward()
        self.assertIsNotNone(model.predictor.output.weight.grad)
        self.assertTrue(torch.isfinite(model.predictor.output.weight.grad).all())

    def test_film_starts_as_the_identity_and_can_read_beta(self):
        from fog_research import FogAdapter
        from ia_yolo import condition_beta

        adapter = FogAdapter()
        image = torch.rand(2, 3, 16, 16)
        beta = torch.tensor([[0.0], [0.02]])
        self.assertTrue(torch.equal(adapter(image, beta), image))
        rows = [{'beta': 0.005}, {'beta': 0.02}]
        self.assertTrue(torch.equal(condition_beta(rows, image, 'film-beta'),
                                    torch.tensor([[0.005], [0.02]])))
        self.assertTrue(torch.equal(condition_beta(rows, image, 'film-zero'),
                                    torch.zeros(2, 1)))


if __name__ == '__main__':
    unittest.main()
