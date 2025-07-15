from typing import Any, List

import numpy as np
import torch

from gigl.common.logger import Logger

logger = Logger()


class CountMinSketch(object):
    """
    A probability data structure that can be used to estimate the frequency of an item in a stream.
    For full details please refer to the paper
    https://dsf.berkeley.edu/cs286/papers/countmin-latin2004.pdf

    There is also a good blog from Redis to talk about its application
    https://redis.io/blog/count-min-sketch-the-art-and-science-of-estimating-stuff/

    How accurate is the estimation?
        Denote the total count is N, and the width of the table is w, the depth of the table is d.
        For each row (in d), we have atleast 1/2 of the probability that the hashed count is less than 2N/w
        Then, with d rows, the final results is more than 2N/w larger than the actual count with a probability less than 1/2^d

    Currently, this implementation only uses single thread, and we might want to optimize it if we see performance issue
    """

    def __init__(self, width: int = 2000, depth: int = 10):
        self.__width: int = width
        self.__depth: int = depth
        self.__table: np.ndarray = np.zeros((depth, width), dtype=np.int32)
        self.__total: int = 0

    def __hash_all(self, item: Any) -> List[int]:
        """
        Return the hash values of the item for all hash functions
        """

        def hash_i(x: Any, i: int) -> int:
            """
            Return the hash value of the item for the i-th hash function
            """
            # Note that python built-in hash function is not deterministic across different processes for many types
            # So we should be careful to only use the CMS in the same process
            return hash((x, i))

        return [hash_i(item, i) for i in range(self.__depth)]

    def add(self, item: Any, delta: int = 1) -> None:
        """
        Add an item to the sketch
        """
        hashed_values: List[int] = self.__hash_all(item)
        for i, hashed_value in enumerate(hashed_values):
            self.__table[i][hashed_value % self.__width] += delta
        self.__total += delta

    def add_torch_long_tensor(self, tensor: torch.LongTensor) -> None:
        """
        Add all items in a torch long tensor to the sketch
        """
        tensor_cpu = tensor.cpu().numpy()
        for item in tensor_cpu:
            self.add(item)

    def total(self) -> int:
        """
        Return the total number of items seen so far
        """
        return self.__total

    def estimate(self, item: Any) -> int:
        """
        Return the estimated count of the item
        """
        hashed_values: List[int] = self.__hash_all(item)
        return min(
            self.__table[i][hashed_value % self.__width]
            for i, hashed_value in enumerate(hashed_values)
        )

    def estimate_torch_long_tensor(self, tensor: torch.LongTensor) -> torch.LongTensor:
        """
        Return the estimated count of all items in a torch long tensor
        """
        tensor_cpu = tensor.cpu().numpy()
        return torch.tensor(  # type: ignore
            [self.estimate(item) for item in tensor_cpu],
            dtype=torch.long,
        )

    def get_table(self) -> np.ndarray:
        """
        Return the internal state of the table, for testing purpose
        """
        return self.__table


def calculate_in_batch_candidate_sampling_probability(
    frequency_tensor: torch.LongTensor, total_cnt: int, batch_size: int
) -> torch.Tensor:
    """
    Calculate in batch negative sampling rate given the frequency tensor, total count and batch size.
    Please see https://www.tensorflow.org/extras/candidate_sampling.pdf for more details
    Here we estimate the negative sampling probability Q(y|x)
    P(candidate in batch | x) ~= P(candidate in batch)
                               = 1 - P(candidate not in batch)
                               = 1 - P(candidate not in any position in batch)
                               ~= 1 - (1 - frequency / total_cnt) ^ batch_size
                               ~= 1 - (1 - batch_size * frequency / total_cnt)
                               = batch_size * frequency / total_cnt
    Where the approximation only holds when frequency / total_cnt << 1, which may not be true at the very beginning of training
    Thus, we cap the probability to be at most 1.0
    Note that the estimation for positive and hard negatives may be less accurate than for random negatives
    because there is a larger error in P(candidate in batch | x) ~= P(candidate in batch)
    """
    estimated_prob: torch.FloatTensor = (
        batch_size * frequency_tensor.float() / total_cnt  # type: ignore
    )
    return estimated_prob.clamp(max=1.0)
