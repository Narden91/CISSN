import logging
import os
import random
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from cissn.data.dataset import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar
from cissn.data.registry import supported_datasets, verify_dataset
from typing import Tuple, Union, Any, Dict
from types import SimpleNamespace

logger = logging.getLogger(__name__)

_verified_datasets: dict[Path, tuple[int, int]] = {}

_DATA_REGISTRY: dict = {
    'ETTh1':         (Dataset_ETT_hour,   'h'),
    'ETTh2':         (Dataset_ETT_hour,   'h'),
    'ETTm1':         (Dataset_ETT_minute, 't'),
    'ETTm2':         (Dataset_ETT_minute, 't'),
    'weather':       (Dataset_Custom,     't'),
    'exchange_rate': (Dataset_Custom,     'd'),
    'ECL':           (Dataset_Custom,     'h'),
    'traffic':       (Dataset_Custom,     'h'),
    'ILI':           (Dataset_Custom,     'w'),
    'solar':         (Dataset_Solar,      't'),
}


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_data_loader(args: Union[SimpleNamespace, Dict[str, Any]], flag: str) -> Tuple[Any, DataLoader]:
    """
    Get data loader for time-series benchmark datasets.

    Args:
        args: Configuration object (Namespace or dict) containing:
              data, root_path, data_path, seq_len, label_len, pred_len,
              features, target, batch_size, freq, num_workers
        flag: Split flag ('train', 'val', 'test', 'pred')

    Returns:
        dataset: The created dataset object
        data_loader: The PyTorch DataLoader
    """
    if isinstance(args, dict):
        args = SimpleNamespace(**args)

    if getattr(args, 'batch_size', 0) <= 0:
        raise ValueError(f"batch_size must be a positive integer; got {getattr(args, 'batch_size', None)}.")

    if args.data not in _DATA_REGISTRY:
        supported = ', '.join(supported_datasets())
        raise ValueError(f"Unknown dataset {args.data!r}. Supported datasets: {supported}.")

    if flag not in {'train', 'val', 'cal', 'test', 'pred'}:
        raise ValueError(f"flag must be one of 'train', 'val', 'cal', 'test', 'pred'; got {flag!r}.")

    if not os.environ.get('CISSN_SKIP_DATA_VERIFY'):
        # Sealed confirmation runs require a byte-exact dataset match, not
        # just a structurally plausible one -- a silently modified file with
        # the same shape would otherwise pass verification.
        require_exact = getattr(args, 'evidence_role', None) == 'confirmation'
        dataset_path = (Path(args.root_path) / args.data_path).resolve()
        if not dataset_path.exists():
            verify_dataset(
                args.data, data_root=getattr(args, 'root_path', None), strict=True,
                require_exact_checksum=require_exact,
            )
        stat = dataset_path.stat()
        fingerprint = (stat.st_size, stat.st_mtime_ns)
        if _verified_datasets.get(dataset_path) != fingerprint:
            verify_dataset(
                args.data, data_root=getattr(args, 'root_path', None), strict=True,
                require_exact_checksum=require_exact,
            )
            _verified_datasets[dataset_path] = fingerprint

    Data, default_freq = _DATA_REGISTRY[args.data]
    freq = getattr(args, 'freq', default_freq) or default_freq

    if flag == 'train':
        shuffle_flag = True
        # Never drop training data. drop_last exists to shield batch-statistics
        # layers from a short final batch, but these models use LayerNorm (which
        # normalises per sample) and a covariance loss computed over
        # batch x seq_len, so even a single-sample batch yields finite,
        # well-conditioned gradients. Dropping the remainder would discard
        # len(split) % batch_size samples every epoch -- 45% of exchange_rate at
        # batch 2048 -- making results incomparable to baselines trained on the
        # full split.
        drop_last = False
        batch_size = args.batch_size
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    else:
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size

    dataset_kwargs = dict(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        cal_fraction=getattr(args, 'cal_fraction', 0.2),
    )
    if issubclass(Data, Dataset_Custom):
        dataset_kwargs['freq'] = freq

    data_set = Data(**dataset_kwargs)

    dataset_length = len(data_set)
    if flag == 'train' and dataset_length < batch_size:
        raise ValueError(
            f"Training split contains only {dataset_length} samples, which is smaller than batch_size={batch_size}."
        )

    logger.debug("%s split: %d samples", flag, dataset_length)
    
    split_seed = {"train": 0, "val": 1, "cal": 2, "test": 3, "pred": 4}[flag]
    generator = torch.Generator().manual_seed(int(getattr(args, "seed", 0)) + split_seed)
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        generator=generator,
        worker_init_fn=_seed_worker if args.num_workers > 0 else None,
    )
    
    return data_set, data_loader
