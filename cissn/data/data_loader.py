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
        
    data_dict = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute,
    }
    
    Data = data_dict[args.data]
    
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target
    )
    
    print(f'{flag}: {len(data_set)}')
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    
    return data_set, data_loader
