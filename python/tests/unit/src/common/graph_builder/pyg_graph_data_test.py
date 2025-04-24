import unittest

import torch

from gigl.common.logger import Logger
from gigl.src.common.graph_builder.pyg_graph_data import PygGraphData

logger = Logger()


class PygGraphDataTest(unittest.TestCase):
    def test_equality(self):
        data = PygGraphData()
        data["1"].x = torch.tensor([[1, 1], [2, 2]])
        data["2"].x = torch.tensor([[3, 3]])
        data["1", "1", "1"].edge_attr = torch.tensor([[1, 2], [1, 3]])
        data["1", "1", "2"].edge_attr = torch.tensor([[1]])
        data["1", "1", "1"].edge_index = torch.LongTensor([[0], [1]])
        data["1", "1", "2"].edge_index = torch.LongTensor([[0, 1], [0, 0]])

        data2 = PygGraphData()
        data2["1"].x = torch.tensor([[1, 1], [2, 2]])
        data2["2"].x = torch.tensor([[3, 3]])
        data2["1", "1", "1"].edge_attr = torch.tensor([[1, 2], [1, 3]])
        data2["1", "1", "2"].edge_attr = torch.tensor([[1]])
        data2["1", "1", "1"].edge_index = torch.LongTensor([[0], [1]])
        data2["1", "1", "2"].edge_index = torch.LongTensor([[0, 1], [0, 0]])

        self.assertEquals(data, data2)

        data = PygGraphData()
        data["1"].x = torch.tensor([[1, 1], [2, 2]])
        data["2"].x = torch.tensor([[3, 3]])
        data["1", "1", "1"].edge_attr = torch.tensor([[1, 2], [1, 3]])
        data["1", "1", "2"].edge_attr = torch.tensor([[1]])
        data["1", "1", "1"].edge_index = torch.LongTensor([[0], [1]])
        data["1", "1", "2"].edge_index = torch.LongTensor([[0, 1], [0, 0]])

        data2 = PygGraphData()
        data2["1"].x = torch.tensor([[1, 1], [2, 2]])
        data2["2"].x = torch.tensor([[3, 3]])

        self.assertNotEquals(data, data2)

        data = PygGraphData()
        data["1"].x = torch.tensor([[1, 1], [2, 2]])
        data["2"].x = torch.tensor([[3, 3]])

        data2 = PygGraphData()
        data2["1"].x = torch.tensor([[1, 1], [2, 2]])
        data2["2"].x = torch.tensor([[3, 3]])

        self.assertEquals(data, data2)

        data = PygGraphData()
        data["1"].x = torch.tensor([[1, 1], [2, 2]])
        data["2"].x = torch.tensor([[3, 3]])

        data2 = PygGraphData()
        data2["1"].x = torch.tensor([[1, 2], [2, 2]])
        data2["2"].x = torch.tensor([[3, 3]])

        self.assertNotEquals(data, data2)
