import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import List, Optional

class BaseETTDataset(Dataset):
    """
    Base class for ETT datasets (Electricity Transformer Temperature).
    Handles data loading, scaling, and time feature extraction.
    """
    def __init__(self, root_path: str, flag: str = 'train', size: Optional[List[int]] = None,
                 features: str = 'S', data_path: str = 'ETTh1.csv',
                 target: str = 'OT', scale: bool = True, **kwargs):
        # size [seq_len, label_len, pred_len]
        if size is None:
            size = [24 * 4 * 4, 24 * 4, 24 * 4]
        if len(size) != 3:
            raise ValueError(f"size must contain [seq_len, label_len, pred_len]; got {size!r}.")

        self.seq_len, self.label_len, self.pred_len = size
        for name, value in (
            ("seq_len", self.seq_len),
            ("label_len", self.label_len),
            ("pred_len", self.pred_len),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer; got {value}.")
        if self.label_len > self.seq_len:
            raise ValueError(
                f"label_len must be <= seq_len for overlapping decoder context; got label_len={self.label_len}, seq_len={self.seq_len}."
            )

        # init ('pred' uses the same split window as test — rolling forecast on held-out tail)
        type_map = {'train': 0, 'val': 1, 'test': 2, 'pred': 2}
        if flag not in type_map:
            raise ValueError(f"flag must be one of {sorted(type_map)}; got {flag!r}.")
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.kwargs = kwargs # Store extra args for custom overrides
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        if 'date' not in df_raw.columns:
            raise ValueError("Dataset must contain a 'date' column for time feature extraction.")

        # Define borders based on dataset type
        # Allow overriding borders via kwargs for custom splits
        if 'borders' in self.kwargs:
             # Expects list of [train_start, val_start, test_start, test_end] or similar structure matched to subclasses
             # But easier: expects tuple (border1s, border2s)
             border1s, border2s = self.kwargs['borders']
        else:
            border1s, border2s = self._get_borders()
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        n_rows = len(df_raw)
        if not (0 <= border1 < border2 <= n_rows):
            raise ValueError(
                f"Invalid split borders for {self.data_path}: border1={border1}, border2={border2}, rows={n_rows}."
            )

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            if self.target not in df_raw.columns:
                raise ValueError(f"Target column {self.target!r} not found in dataset columns.")
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"features must be one of 'M', 'MS', or 'S'; got {self.features!r}.")

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        
        # Time features
        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        self.data_stamp = self._extract_time_features(df_stamp)
        self.length = len(self.data_x) - self.seq_len - self.pred_len + 1
        if self.length <= 0:
            raise ValueError(
                "Dataset split is too short for the requested window sizes: "
                f"split_length={len(self.data_x)}, seq_len={self.seq_len}, pred_len={self.pred_len}."
            )

    def _get_borders(self):
        """
        Define train/val/test split borders.
        Should be implemented/overridden by subclasses.
        """
        raise NotImplementedError

    def _extract_time_features(self, df_stamp):
        """
        Extract time features from date column.
        Should be implemented/overridden by subclasses.
        """
        raise NotImplementedError

    def __getitem__(self, index):
        if index < 0 or index >= self.length:
            raise IndexError(f"Index {index} is out of range for dataset of length {self.length}.")

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        expected_x = self.seq_len
        expected_y = self.label_len + self.pred_len
        if len(seq_x) != expected_x or len(seq_x_mark) != expected_x:
            raise RuntimeError(
                f"Encoder window extraction failed at index {index}: expected {expected_x} rows, got data={len(seq_x)} marks={len(seq_x_mark)}."
            )
        if len(seq_y) != expected_y or len(seq_y_mark) != expected_y:
            raise RuntimeError(
                f"Decoder window extraction failed at index {index}: expected {expected_y} rows, got data={len(seq_y)} marks={len(seq_y_mark)}."
            )

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.length

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour(BaseETTDataset):
    def _get_borders(self):
        # 12 months train, 4 months val, 4 months test (hourly data)
        # 30 days * 24 hours = 720 hours/month
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        return border1s, border2s

    def _extract_time_features(self, df_stamp):
        # Simple time encoding: Month, Day, Weekday, Hour
        data_stamp = np.zeros((len(df_stamp), 4))
        data_stamp[:, 0] = df_stamp.date.dt.month.values / 12.0 - 0.5
        data_stamp[:, 1] = df_stamp.date.dt.day.values / 31.0 - 0.5
        data_stamp[:, 2] = df_stamp.date.dt.dayofweek.values / 7.0 - 0.5
        data_stamp[:, 3] = df_stamp.date.dt.hour.values / 24.0 - 0.5
        return data_stamp


class Dataset_ETT_minute(BaseETTDataset):
    def __init__(self, root_path: str, flag: str = 'train', size: Optional[List[int]] = None,
                 features: str = 'S', data_path: str = 'ETTm1.csv',
                 target: str = 'OT', scale: bool = True, **kwargs):
        super().__init__(root_path, flag, size, features, data_path, target, scale, **kwargs)

    def _get_borders(self):
        # 12 months train, 4 months val, 4 months test (minute data - 15min intervals usually for ETTm)
        # But here logic multiplies by 4, likely assuming 15-min intervals (4 per hour) if index is just counts
        # ETTm1 is 15-minute level. 24 * 4 points per day.
        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        return border1s, border2s

    def _extract_time_features(self, df_stamp):
        # Time encoding: Month, Day, Weekday, Hour, Minute
        data_stamp = np.zeros((len(df_stamp), 5))
        data_stamp[:, 0] = df_stamp.date.dt.month.values / 12.0 - 0.5
        data_stamp[:, 1] = df_stamp.date.dt.day.values / 31.0 - 0.5
        data_stamp[:, 2] = df_stamp.date.dt.dayofweek.values / 7.0 - 0.5
        data_stamp[:, 3] = df_stamp.date.dt.hour.values / 24.0 - 0.5
        data_stamp[:, 4] = df_stamp.date.dt.minute.values / 60.0 - 0.5
        return data_stamp
