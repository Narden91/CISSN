import torch
from torch.utils.data import DataLoader
from cissn.data.dataset import Dataset_ETT_hour, Dataset_ETT_minute
from typing import Tuple, Union, Any, Dict
from types import SimpleNamespace

def get_data_loader(args: Union[SimpleNamespace, Dict[str, Any]], flag: str) -> Tuple[Any, DataLoader]:
    """
    Get data loader for ETT datasets.
    
    Args:
        args: Configuration object (Namespace or dict) containing:
              data, root_path, data_path, seq_len, label_len, pred_len,
              features, target, batch_size, freq, num_workers
        flag: Split flag ('train', 'val', 'test', 'pred')
        
    Returns:
        dataset: The created dataset object
        data_loader: The PyTorch DataLoader
    """
    # Normalize args to object access if dict
    if isinstance(args, dict):
        args = SimpleNamespace(**args)

    if getattr(args, 'batch_size', 0) <= 0:
        raise ValueError(f"batch_size must be a positive integer; got {getattr(args, 'batch_size', None)}.")
        
    data_dict = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute,
    }

    if args.data not in data_dict:
        supported = ', '.join(sorted(data_dict))
        raise ValueError(f"Unknown dataset {args.data!r}. Supported datasets: {supported}.")

    if flag not in {'train', 'val', 'test', 'pred'}:
        raise ValueError(f"flag must be one of 'train', 'val', 'test', 'pred'; got {flag!r}.")
    
    Data = data_dict[args.data]
    
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

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target
    )

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
