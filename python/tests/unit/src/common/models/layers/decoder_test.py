import unittest

import torch
import torch.nn.functional as F

from gigl.src.common.models.layers.decoder import DecoderType, LinkPredictionDecoder


class TestLinkPredictionDecoder(unittest.TestCase):
    def setUp(self):
        self.model = LinkPredictionDecoder(
            decoder_type=DecoderType.hadamard_MLP, decoder_channel_list=[4, 2, 1]
        )
        self.model1 = LinkPredictionDecoder(
            decoder_type=DecoderType.inner_product, decoder_channel_list=[4, 2, 1]
        )
        torch.manual_seed(0)
        self.query_embeddings = F.normalize(
            torch.tensor(
                [
                    [1.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.9, 1.1],
                    [0.0, 0.0, 0.9, 1.1],
                    [0.0, 0.0, 0.1, 1.1],
                ]
            ),
            p=2,
            dim=1,
        )
        self.candidate_embeddings = torch.tensor(
            [
                [0.9806, 0.1961, 0.0000, 0.0000],
                [0.2822, 0.9407, 0.1881, 0.0000],
                [0.0000, 0.0000, 0.9285, 0.3714],
                [0.0000, 0.0000, 0.4061, 0.9138],
                [0.4661, 0.4883, 0.5105, 0.5327],
                [0.5327, 0.5105, 0.4883, 0.4661],
            ]
        )

    def testForward(self):
        self.assertEqual(
            self.model.forward(
                self.query_embeddings, self.candidate_embeddings
            ).shape.numel(),
            5 * 6,
        )

        actual = self.model1.forward(self.query_embeddings, self.candidate_embeddings)
        expected = torch.tensor(
            [
                [0.8321, 0.8647, 0.0000, 0.0000, 0.6749, 0.7377],
                [0.9806, 0.2822, 0.0000, 0.0000, 0.4661, 0.5327],
                [0.0000, 0.1191, 0.8754, 0.9644, 0.7356, 0.6700],
                [0.0000, 0.1191, 0.8754, 0.9644, 0.7356, 0.6700],
                [0.0000, 0.0170, 0.4539, 0.9468, 0.5767, 0.5084],
            ],
            dtype=torch.float32,
        )

        self.assertTrue(torch.allclose(expected, actual, atol=1e-4))

    def testExceptions(self):
        with self.assertRaises(AttributeError):
            LinkPredictionDecoder(
                decoder_type=DecoderType.outer_product,  # type: ignore
                decoder_channel_list=None,  # decoder_type is incorrect
            )
        with self.assertRaises(ValueError):
            LinkPredictionDecoder(
                decoder_type=DecoderType.hadamard_MLP,
                decoder_channel_list=None,  # decoder_channel_list not specified for Hadamard MLP decoder
            )
        with self.assertRaises(ValueError):
            LinkPredictionDecoder(
                decoder_type=DecoderType.hadamard_MLP,
                decoder_channel_list=[
                    1
                ],  # decoder_channel_list has length <=1 for Hadamard MLP decoder
            )
        with self.assertRaises(ValueError):
            LinkPredictionDecoder(
                decoder_type=DecoderType.hadamard_MLP,
                decoder_channel_list=[2, 2],  # decoder_channel_list last element != 1
            )
