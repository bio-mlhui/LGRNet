
import math
import torch.distributed as dist
from typing import TypeVar, Optional, Iterator        
T_co = TypeVar('T_co', covariant=True)
from torch.utils.data.distributed import DistributedSampler
import detectron2.utils.comm as comm
from utils.misc import all_gather
from torch.utils.data import Sampler
import torch

import logging
import itertools

class TrainRandomSampler_ByEpoch(Sampler[int]):
    def __init__(self, 
                 data_source,
                 seed,
                 ) -> None:
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.seed = seed
        self.epoch = None

    def __iter__(self):
        seed = self.seed + self.epoch
        print(f'generating a new indices permutations for this epoch using seed {seed}')
        n = len(self.data_source)
        g = torch.Generator()
        g.manual_seed(seed)
        
        for _ in range(self.num_samples // n):
            yield from torch.randperm(n, generator=g).tolist()
        yield from torch.randperm(n, generator=g).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
  
  

class Train_InfiniteSampler_Distributed(Sampler[T_co]):
    def __init__(self, 
                 inf_stream_fn, 
                 start_idx: int = 0,
                 end_idx = None,
                 
                ): 
        self.rank = comm.get_rank()
        self.num_replicas = comm.get_world_size()
        self.start_idx = start_idx 
        self.end_idx = end_idx
        self.inf_stream_fn = inf_stream_fn

    def set_iter_first_sample_idx(self, idx):
        self.start_idx = idx

    def set_iter_last_sample_idx(self, idx):
        self.end_idx = idx

    def __iter__(self) -> Iterator[T_co]:
        logging.debug(f'在 infinite stream 上定位到{self.start_idx} 为开头')
        yield from itertools.islice(self.inf_stream_fn(), self.start_idx + self.rank, self.end_idx, self.num_replicas)

class Evaluate_ExactSampler_Distributed(Sampler[T_co]):
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.rank = comm.get_rank()
        self.num_replicas = comm.get_world_size()
        indices = list(range(len(self.dataset)))  
        self.indices = indices[self.rank:len(self.dataset):self.num_replicas]


    def __iter__(self):
        yield from self.indices

    def __len__(self):
        return len(self.indices)


class TrainRandomSampler_ByEpoch_Distributed(Sampler[T_co]):
    def __init__(self, 
                 dataset, num_replicas,
                 rank,
                 seed: int = 0) -> None:
        if rank >= num_replicas or rank < 0:
            raise ValueError("Invalid rank {}, rank should be in the interval"" [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = None

        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        seed = self.seed + self.epoch
        logging.debug(f'generating a new indices permutations for this epoch using seed {seed}')
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        # add extra samples to make it evenly divisible
        padding_size = self.total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        assert len(indices) == self.total_size
        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        self.epoch = None 
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class InferenceSampler(Sampler):
    """
    Produce indices for inference across all workers.
    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """

    def __init__(self, size: int):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
        """
        self._size = size
        assert size > 0
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[: rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)