import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple


class BaseETTDataset(Dataset):
    """
    Base class for ETT datasets (Electricity Transformer Temperature).
    Handles data loading, scaling, and time feature extraction.
    """

    _DEFAULT_DATA_PATH: str = "ETTh1.csv"

    def __init__(
        self,
        root_path: str,
        flag: str = "train",
        size: Optional[List[int]] = None,
        features: str = "S",
        data_path: str = "",
        target: str = "OT",
        scale: bool = True,
        **kwargs,
    ):
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

        if flag not in {"train", "val", "cal", "test", "pred"}:
            raise ValueError(f"flag must be one of ['cal', 'pred', 'test', 'train', 'val']; got {flag!r}.")
        self.flag = flag
        self.set_type = 0

        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path or self._DEFAULT_DATA_PATH
        self.kwargs = kwargs
        self._read_data()

    def _load_raw_dataframe(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.root_path, self.data_path))

    def _resolve_split_index(self, n_splits: int) -> int:
        if n_splits == 4:
            return {"train": 0, "val": 1, "cal": 2, "test": 3, "pred": 3}[self.flag]
        if n_splits == 3:
            return {"train": 0, "val": 1, "cal": 1, "test": 2, "pred": 2}[self.flag]
        raise ValueError(f"Expected 3 or 4 chronological split borders; got {n_splits}.")

    def _select_data_columns(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        data_columns = list(df_raw.columns[1:])

        if self.features == "M":
            return df_raw[data_columns]
        if self.features == "MS":
            if self.target not in data_columns:
                raise ValueError(f"Target column {self.target!r} not found in dataset columns.")
            ordered_columns = [col for col in data_columns if col != self.target] + [self.target]
            return df_raw[ordered_columns]
        if self.features == "S":
            if self.target not in df_raw.columns:
                raise ValueError(f"Target column {self.target!r} not found in dataset columns.")
            return df_raw[[self.target]]
        raise ValueError(f"features must be one of 'M', 'MS', or 'S'; got {self.features!r}.")

    def _read_data(self):
        self.scaler = StandardScaler()
        df_raw = self._load_raw_dataframe()
        if "date" not in df_raw.columns:
            raise ValueError("Dataset must contain a 'date' column for time feature extraction.")

        if "borders" in self.kwargs:
            border1s, border2s = self.kwargs["borders"]
        else:
            border1s, border2s = self._get_borders(df_raw)

        if len(border1s) != len(border2s):
            raise ValueError("Split border start/end lists must have the same length.")
        self.set_type = self._resolve_split_index(len(border1s))

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        n_rows = len(df_raw)
        if not (0 <= border1 < border2 <= n_rows):
            raise ValueError(
                f"Invalid split borders for {self.data_path}: border1={border1}, border2={border2}, rows={n_rows}."
            )

        df_data = self._select_data_columns(df_raw)

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # data_x and data_y slice the same array.  They are kept separate so
        # that future subclasses can override one (e.g. multi-target setups).
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        df_stamp = df_raw[["date"]][border1:border2].copy()
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        self.data_stamp = self._extract_time_features(df_stamp)

        self.length = len(self.data_x) - self.seq_len - self.pred_len + 1
        if self.length <= 0:
            raise ValueError(
                "Dataset split is too short for the requested window sizes: "
                f"split_length={len(self.data_x)}, seq_len={self.seq_len}, pred_len={self.pred_len}."
            )

    @staticmethod
    def _build_time_features(
        df_stamp: pd.DataFrame,
        fields: List[Tuple[str, float]],
    ) -> np.ndarray:
        """
        Build a normalized time-feature array.

        Args:
            df_stamp: DataFrame with a 'date' datetime column.
            fields: List of (accessor_name, normalizer_period) pairs.
                    Normalization: value / period - 0.5
                    Accessor names map to pd.DatetimeIndex attributes,
                    e.g. 'month', 'day', 'dayofweek', 'hour', 'minute'.
                    For ISO week use 'isocalendar_week'.
        Returns:
            np.ndarray of shape (len(df_stamp), len(fields))
        """
        data = np.zeros((len(df_stamp), len(fields)))
        for col, (accessor, period) in enumerate(fields):
            if accessor == "isocalendar_week":
                values = df_stamp.date.dt.isocalendar().week.values.astype(float)
            else:
                values = getattr(df_stamp.date.dt, accessor).values.astype(float)
            data[:, col] = values / period - 0.5
        return data

    def _get_borders(self, df_raw: pd.DataFrame):
        """
        Define train/val/test split borders using a 12/4/4-month convention.
        Used by ETT hourly and minute subclasses. Override for custom splits.
        """
        dates = pd.to_datetime(df_raw["date"])
        start_date = dates.iloc[0]
        train_end = start_date + pd.DateOffset(months=12)
        val_end = train_end + pd.DateOffset(months=2)
        cal_end = val_end + pd.DateOffset(months=2)
        test_end = cal_end + pd.DateOffset(months=4)
        train_n = int((dates < train_end).sum())
        val_n = int((dates < val_end).sum())
        cal_n = int((dates < cal_end).sum())
        test_n = int((dates < test_end).sum())
        border1s = [0, train_n - self.seq_len, val_n - self.seq_len, cal_n - self.seq_len]
        border2s = [train_n, val_n, cal_n, test_n]
        return border1s, border2s

    def _extract_time_features(self, df_stamp: pd.DataFrame) -> np.ndarray:
        """Extract time features from date column. Override in subclasses."""
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
                f"Encoder window extraction failed at index {index}: expected {expected_x} rows, "
                f"got data={len(seq_x)} marks={len(seq_x_mark)}."
            )
        if len(seq_y) != expected_y or len(seq_y_mark) != expected_y:
            raise RuntimeError(
                f"Decoder window extraction failed at index {index}: expected {expected_y} rows, "
                f"got data={len(seq_y)} marks={len(seq_y_mark)}."
            )

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.length

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


_HOURLY_FIELDS = [("month", 12.0), ("day", 31.0), ("dayofweek", 7.0), ("hour", 24.0)]
_MINUTE_FIELDS = _HOURLY_FIELDS + [("minute", 60.0)]


class Dataset_ETT_hour(BaseETTDataset):
    _DEFAULT_DATA_PATH = "ETTh1.csv"

    def _extract_time_features(self, df_stamp: pd.DataFrame) -> np.ndarray:
        return self._build_time_features(df_stamp, _HOURLY_FIELDS)


class Dataset_ETT_minute(BaseETTDataset):
    _DEFAULT_DATA_PATH = "ETTm1.csv"

    def _extract_time_features(self, df_stamp: pd.DataFrame) -> np.ndarray:
        return self._build_time_features(df_stamp, _MINUTE_FIELDS)


_DAILY_FIELDS = [("month", 12.0), ("day", 31.0), ("dayofweek", 7.0)]
_WEEKLY_FIELDS = [("month", 12.0), ("isocalendar_week", 52.0)]


class Dataset_Custom(BaseETTDataset):
    """
    Generic dataset class for non-ETT benchmark datasets (Weather, Exchange-Rate,
    Electricity/ECL, Traffic, ILI, Solar-Energy, etc.).

    Split policy: 60% train / 10% validation / 10% calibration / 20% test.

    Args:
        freq: Sampling frequency string used to select time features.
              'h' (hourly, default), 't'/'10min'/'15min' (sub-hourly),
              'd' (daily), 'w' (weekly).
    """

    def __init__(
        self,
        root_path: str,
        flag: str = "train",
        size: Optional[List[int]] = None,
        features: str = "S",
        data_path: str = "weather.csv",
        target: str = "OT",
        scale: bool = True,
        freq: str = "h",
        **kwargs,
    ):
        self.freq = freq
        super().__init__(root_path, flag, size, features, data_path, target, scale, **kwargs)

    def _get_borders(self, df_raw: pd.DataFrame):
        n = len(df_raw)
        num_train = int(n * 0.6)
        num_vali = int(n * 0.1)
        num_cal = int(n * 0.1)
        num_test = int(n * 0.2)
        test_start = n - num_test
        cal_start = test_start - num_cal
        val_start = cal_start - num_vali
        border1s = [0, val_start - self.seq_len, cal_start - self.seq_len, test_start - self.seq_len]
        border2s = [val_start, cal_start, test_start, n]
        return border1s, border2s

    def _extract_time_features(self, df_stamp: pd.DataFrame) -> np.ndarray:
        freq = self.freq.lower()
        if freq in ("t", "min", "10min", "15min"):
            fields = _MINUTE_FIELDS
        elif freq in ("d", "daily", "1d"):
            fields = _DAILY_FIELDS
        elif freq in ("w", "weekly", "1w"):
            fields = _WEEKLY_FIELDS
        else:
            fields = _HOURLY_FIELDS
        return self._build_time_features(df_stamp, fields)


class Dataset_Solar(Dataset_Custom):
    """Solar-Energy benchmark loader for the raw LSTNet `solar_AL.txt` file.

    The source file has no date column or header. We create a deterministic
    10-minute synthetic index and name the final channel `OT` so `S`/`MS`
    target selection behaves like the other benchmark CSV files.
    """

    def __init__(
        self,
        root_path: str,
        flag: str = "train",
        size: Optional[List[int]] = None,
        features: str = "S",
        data_path: str = "solar_AL.txt",
        target: str = "OT",
        scale: bool = True,
        freq: str = "t",
        **kwargs,
    ):
        super().__init__(root_path, flag, size, features, data_path, target, scale, freq, **kwargs)

    def _load_raw_dataframe(self) -> pd.DataFrame:
        path = os.path.join(self.root_path, self.data_path)
        df = pd.read_csv(path, header=None)
        if df.empty:
            raise ValueError(f"Solar dataset at {path!r} is empty.")
        columns = [str(i) for i in range(df.shape[1] - 1)] + [self.target]
        df.columns = columns
        date = pd.DataFrame({"date": pd.date_range("2006-01-01", periods=len(df), freq="10min")})
        return pd.concat([date, df], axis=1)
