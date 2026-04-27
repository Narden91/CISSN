import torch
from torch.utils.data import DataLoader
from cissn.data.dataset import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from typing import Tuple, Union, Any, Dict
from types import SimpleNamespace

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
    'solar':         (Dataset_Custom,     't'),
}

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
        supported = ', '.join(sorted(_DATA_REGISTRY))
        raise ValueError(f"Unknown dataset {args.data!r}. Supported datasets: {supported}.")

    if flag not in {'train', 'val', 'test', 'pred'}:
        raise ValueError(f"flag must be one of 'train', 'val', 'test', 'pred'; got {flag!r}.")

    Data, default_freq = _DATA_REGISTRY[args.data]
    freq = getattr(args, 'freq', default_freq) or default_freq

    if flag == 'train':
        shuffle_flag = True
        drop_last = True
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
    )
    if Data is Dataset_Custom:
        dataset_kwargs['freq'] = freq

    data_set = Data(**dataset_kwargs)

    dataset_length = len(data_set)
    if flag == 'train' and dataset_length < batch_size:
        raise ValueError(
            f"Training split contains only {dataset_length} samples, which is smaller than batch_size={batch_size}."
        )
    
    print(f'{flag}: {dataset_length}')
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    
    return data_set, data_loader
