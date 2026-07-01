# step 0

import os
import math
import random
import warnings
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
from IPython.display import display

import types
import matplotlib
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CSV_PATH = Path("data") / "DecorderOnlyS&P.csv"

# step 1

def load_ohlcv_csv(path: str, add_tech_indicators: bool = False) -> pd.DataFrame:
    logger.info(f"Loading CSV: {path}")
    df = pd.read_csv(path, sep="|")

    df = df.loc[:, ~df.columns.astype(str).str.fullmatch(r"\s*")]
    alias_map = {
        "date": "Date", "timestamp": "Date", "time": "Date",
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "adj close": "Close", "adj_close": "Close", "adjclose": "Close",
        "volume": "Volume", "vol": "Volume",
    }
    lower_to_orig = {c.lower(): c for c in df.columns}
    rename_dict = {lower_to_orig[k]: v for k, v in alias_map.items() if k in lower_to_orig}
    df = df.rename(columns=rename_dict)

    need = ["Date", "Open", "High", "Low", "Close"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}; available columns: {list(df.columns)}")

    if "Volume" not in df.columns:
        logger.warning("CSV does not contain Volume column. Default value 0 is used.")
        df["Volume"] = 0.0
    need.append("Volume")

    n0 = len(df)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=False)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=need)
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    invalid = (
        (df["Open"] <= 0) | (df["High"] <= 0) | (df["Low"] <= 0) | (df["Close"] <= 0) |
        (df["Volume"] < 0) | (df["High"] < df["Low"]) |
        (df["Open"] > df["High"]) | (df["Open"] < df["Low"]) |
        (df["Close"] > df["High"]) | (df["Close"] < df["Low"])
    )
    if invalid.any():
        logger.warning(f"Found {invalid.sum()} invalid price or volume records. Removed.")
        df = df[~invalid].reset_index(drop=True)

    n1 = len(df)
    if n1 != n0:
        logger.info(f"Remaining records after cleaning: {n1} records; removed {n0 - n1} records.")

    if add_tech_indicators:
        try:
            from ta import add_all_ta_features
            df = add_all_ta_features(
                df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
            )
            logger.info("Technical indicators added.")
        except ImportError:
            raise ImportError("Package 'ta' is required to add technical indicators.")

    return df


def add_relative_targets(df: pd.DataFrame, base_col: str = "Close", method: str = "ratio") -> pd.DataFrame:
    raise NotImplementedError(
        "Target construction is omitted in the public version."
    )


def apply_price_features(df):
    raise NotImplementedError(
        "Feature construction is omitted in the public version."
    )


@dataclass
class Scalers:
    x_scaler: StandardScaler
    y_scalers: Dict[str, StandardScaler] = field(default_factory=dict)
    x_cols: List[str] = field(default_factory=lambda: ["Open", "High", "Low", "Close", "Volume"])
    y_cols: List[str] = field(default_factory=lambda: ["High", "Low", "Close"])
    train_minmax: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    dataset_minmax: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    target_mode: str = "absolute"
    relative_def: str = "ratio"
    base_col: str = "Close"

    def update_x_cols(self, df: pd.DataFrame):
        deny = set(["Date", "High_rel", "Low_rel", "Close_rel", "Open", "High", "Low", "Close"])
        base = ["feature_y1", "feature_y2", "feature_y3", "feature_y4", "Volume"]
        xcols = [c for c in base if c in df.columns]
        extras = [c for c in df.columns if c not in deny and c not in xcols]
        self.x_cols = xcols + extras
        logger.info(f"Updated x_cols: {len(self.x_cols)} columns: {self.x_cols[:8]}...")

    def inverse_transform_y(self, Y: np.ndarray, col: str) -> np.ndarray:
        if col not in self.y_scalers:
            raise ValueError(f"Unknown y column: {col}")
        scaler = self.y_scalers[col]
        orig_shape = Y.shape
        if Y.ndim == 3:
            Y2 = Y.reshape(-1, 1)
        elif Y.ndim == 2:
            Y2 = Y
        elif Y.ndim == 1:
            Y2 = Y.reshape(-1, 1)
        else:
            raise ValueError(f"Unsupported Y dimension: {Y.ndim}")
        Y_inv = scaler.inverse_transform(Y2)
        return Y_inv.reshape(orig_shape)

    def relative_to_raw(self, Y_rel: np.ndarray, base: np.ndarray) -> np.ndarray:
        if self.relative_def == "ratio":
            return base * (1.0 + Y_rel)
        elif self.relative_def == "diff":
            return base + Y_rel
        else:
            raise ValueError("relative_def must be either 'ratio' or 'diff'.")


def fit_scalers(train_df: pd.DataFrame, scalers: Optional[Scalers] = None) -> Scalers:
    if scalers is None:
        scalers = Scalers(StandardScaler())

    df_log = train_df.copy()
    df_log["Volume"] = np.log1p(df_log["Volume"])
    scalers.update_x_cols(train_df)

    x_data = df_log[scalers.x_cols].values
    scalers.x_scaler.fit(x_data)

    if scalers.target_mode == "absolute":
        src_high = train_df["High"].values.reshape(-1, 1)
        src_low = train_df["Low"].values.reshape(-1, 1)
        src_close = train_df["Close"].values.reshape(-1, 1)
    else:
        need_cols = ["High_rel", "Low_rel", "Close_rel"]
        if any(c not in train_df.columns for c in need_cols):
            raise ValueError("Relative target mode requires add_relative_targets() before fitting scalers.")
        src_high = train_df["High_rel"].values.reshape(-1, 1)
        src_low = train_df["Low_rel"].values.reshape(-1, 1)
        src_close = train_df["Close_rel"].values.reshape(-1, 1)

    scalers.y_scalers["High"] = StandardScaler().fit(src_high)
    scalers.y_scalers["Low"] = StandardScaler().fit(src_low)
    scalers.y_scalers["Close"] = StandardScaler().fit(src_close)

    for k in ["High", "Low", "Close"]:
        sc = scalers.y_scalers[k]
        logger.info(f"Fitted y_scaler[{k}] mean={sc.mean_[0]:.6f}, std={sc.scale_[0]:.6f}")

    return scalers


def apply_scalers(df: pd.DataFrame, scalers: Scalers) -> Tuple[np.ndarray, np.ndarray]:
    df_log = df.copy()
    df_log["Volume"] = np.log1p(df_log["Volume"])
    X = scalers.x_scaler.transform(df_log[scalers.x_cols].values).astype(np.float32)

    if scalers.target_mode == "absolute":
        y_src = df[["High", "Low", "Close"]].values
    else:
        need = ["High_rel", "Low_rel", "Close_rel"]
        if any(c not in df.columns for c in need):
            raise ValueError("Relative target mode requires High_rel, Low_rel, and Close_rel columns.")
        y_src = df[["High_rel", "Low_rel", "Close_rel"]].values

    y_high_z = scalers.y_scalers["High"].transform(y_src[:, [0]])
    y_low_z = scalers.y_scalers["Low"].transform(y_src[:, [1]])
    y_close_z = scalers.y_scalers["Close"].transform(y_src[:, [2]])
    Y = np.concatenate([y_high_z, y_low_z, y_close_z], axis=1).astype(np.float32)

    logger.info(f"apply_scalers: X={X.shape}, Y={Y.shape}, target_dim=3: High/Low/Close")
    return X, Y


class OHLCVDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, L: int, H: int, dates: Optional[pd.Series] = None):
        super().__init__()
        assert X.ndim == 2 and Y.ndim == 2 and X.shape[0] == Y.shape[0]
        self.X, self.Y, self.L, self.H = X, Y, int(L), int(H)
        self.dates = dates.reset_index(drop=True) if dates is not None else None
        N = len(X)
        self.indices = list(range(self.L - 1, N - self.H))
        if not self.indices:
            raise ValueError(f"No valid sequence found. L={L}, H={H}, N={N}")
        logger.info(f"OHLCVDataset: L={L}, H={H}, samples={len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        t = self.indices[idx]
        x_hist = self.X[t - self.L + 1: t + 1]
        y_seq = self.Y[t + 1: t + 1 + self.H]
        return torch.from_numpy(x_hist), torch.from_numpy(y_seq)


def train_val_test_split(
    df: pd.DataFrame,
    train_ratio=0.7,
    val_ratio=0.15,
    date_splits: Optional[pd.Timestamp] = None,
    min_samples_per_set: Optional[int] = None
):
    assert 0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1

    if date_splits:
        train_end, val_end = pd.to_datetime(date_splits[0]), pd.to_datetime(date_splits[1])
        train_df = df[df["Date"] <= train_end].copy()
        val_df = df[(df["Date"] > train_end) & (df["Date"] <= val_end)].copy()
        test_df = df[df["Date"] > val_end].copy()
    else:
        N = len(df)
        n_train = int(N * train_ratio)
        n_val = int(N * val_ratio)
        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train:n_train + n_val].copy()
        test_df = df.iloc[n_train + n_val:].copy()

    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        assert len(d) > 0, f"{name} split is empty."
        assert d["Date"].is_monotonic_increasing, f"{name} dates are not monotonically increasing."

    assert train_df["Date"].max() < val_df["Date"].min(), "train/val boundary overlap or order error."
    assert val_df["Date"].max() < test_df["Date"].min(), "val/test boundary overlap or order error."

    if min_samples_per_set:
        for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if len(d) < min_samples_per_set:
                raise ValueError(f"{name} set has insufficient samples: {len(d)}")

    logger.info(f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


def make_dataloaders(
    csv_path: str,
    L: int,
    H: int,
    batch_size=64,
    num_workers=None,
    add_tech_indicators=False,
    date_splits=None,
    target_mode: str = "relative",
    relative_def: str = "ratio"
):
    df = load_ohlcv_csv(csv_path, add_tech_indicators=add_tech_indicators)

    if target_mode == "relative":
        df = add_relative_targets(df, base_col="Close", method=relative_def)
        df = df.dropna(subset=["High_rel", "Low_rel", "Close_rel"]).reset_index(drop=True)

    min_samples = L + H
    train_df, val_df, test_df = train_val_test_split(df, 0.7, 0.15, date_splits, min_samples)

    train_df = apply_price_features(train_df)
    val_df = apply_price_features(val_df)
    test_df = apply_price_features(test_df)

    scalers = Scalers(StandardScaler(), target_mode=target_mode, relative_def=relative_def, base_col="Close")
    scalers.y_cols = ["High", "Low", "Close"]

    scalers = fit_scalers(train_df, scalers)

    X_train, Y_train = apply_scalers(train_df, scalers)
    X_val, Y_val = apply_scalers(val_df, scalers)
    X_test, Y_test = apply_scalers(test_df, scalers)

    scalers.train_minmax = {
        "High": (float(train_df["High"].min()), float(train_df["High"].max())),
        "Low": (float(train_df["Low"].min()), float(train_df["Low"].max()))
    }
    scalers.dataset_minmax = {
        "High": (float(df["High"].min()), float(df["High"].max())),
        "Low": (float(df["Low"].min()), float(df["Low"].max()))
    }

    ds_train = OHLCVDataset(X_train, Y_train, L, H, dates=train_df["Date"])
    ds_val = OHLCVDataset(X_val, Y_val, L, H, dates=val_df["Date"])
    ds_test = OHLCVDataset(X_test, Y_test, L, H, dates=test_df["Date"])

    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 4)

    bs = min(batch_size, len(ds_train)) if len(ds_train) > 0 else 1
    pin = torch.cuda.is_available()

    dl_train = DataLoader(
        ds_train,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin
    )

    return dl_train, dl_val, dl_test, scalers

# step 2

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        raise NotImplementedError(
            "Positional encoding implementation is omitted in the public version."
        )

    def _build_pe(self, max_len: int, d_model: int):
        raise NotImplementedError(
            "Positional encoding implementation is omitted in the public version."
        )

    def add_pe(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Positional encoding implementation is omitted in the public version."
        )


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        raise NotImplementedError(
            "Decoder block implementation is omitted in the public version."
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Decoder block implementation is omitted in the public version."
        )


def causal_mask(T: int, device: torch.device) -> torch.Tensor:
    raise NotImplementedError(
        "Causal mask implementation is omitted in the public version."
    )


class DecoderOnlyTSModel(nn.Module):
    def __init__(
        self,
        d_in: int = 5,
        d_out: int = 3,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 2048
    ):
        super().__init__()
        raise NotImplementedError(
            "Model architecture is omitted in the public version."
        )

    def extend_max_len(self, new_max_len: int):
        raise NotImplementedError(
            "Position length extension is omitted in the public version."
        )

    def reset_parameters(self):
        raise NotImplementedError(
            "Parameter initialization is omitted in the public version."
        )

    def forward(self, hist_feat: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Forward pass is omitted in the public version."
        )

    @torch.no_grad()
    def predict(self, hist_feat: torch.Tensor, H: int) -> torch.Tensor:
        raise NotImplementedError(
            "Autoregressive prediction is omitted in the public version."
        )
    
# step 3: Transformer decoder model


# step 4: Load pretrained artifacts and rebuild evaluation data

def load_pretrained_artifacts(*args, **kwargs):
    raise NotImplementedError(
        "Pretrained model loading is omitted in the public version."
    )


def rebuild_evaluation_dataloaders(*args, **kwargs):
    raise NotImplementedError(
        "Evaluation dataloader reconstruction is omitted in the public version."
    )

# step 5: Backtesting and performance evaluation

EVAL_SPLIT = "test"
SELECT_BY = "fill in data"
FINAL_STRATEGY = "fill in data"
FINAL_K = "fill in data"

HOLDS = ["fill in data"]

ENTRY_LAG = 1
ENTRY_WHERE = "fill in data"
EXIT_WHERE = "close"
MIN_HOLD_DAYS_LIST = ["fill in data"]
MIN_HOLD_DAYS = "fill in data"
ALLOW_OVERLAP = "fill in data"

INITIAL_CAPITAL = 100_000.0
FEE_BPS = "fill in data"
SLIPPAGE_BPS = "fill in data"
RISK_FREE_RATE = "fill in data"

PLOT_WHAT = "both"

TP_BUFFER_BPS = 10
SL_BUFFER_BPS = 10
SAME_DAY_PRIORITY = "sl"

COST_MODEL = "static"
POSITION_SIZING = "all_in"
RISK_PER_TRADE_PCT = "fill in data"

USE_POS_SCALE = "fill in data"
POS_SCALE_MAX = "fill in data"

VALID_STRATEGY_NAMES = {
    "fill in data"
}

OOS_STRESS_TEST = "fill in data"
OOS_START = "fill in data"
OOS_END = "fill in data"


@dataclass
class TradePlan:
    entry_idx: int
    exit_idx: int
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    holding_days: int
    horizon_limit: int
    reason: str
    ret_pct: float


def standardize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c: str(c).strip().lower() for c in df.columns}

    def pick_column(candidates):
        for c in df.columns:
            if lower[c] in candidates:
                return c
        for c in df.columns:
            for k in candidates:
                if k in lower[c]:
                    return c
        return None

    c_date = pick_column({"date", "datetime", "timestamp", "time"})
    c_open = pick_column({"open"})
    c_high = pick_column({"high"})
    c_low = pick_column({"low"})
    c_close = pick_column({"close", "adj close", "adj_close", "close*"})
    c_volume = pick_column({"volume", "vol"})

    out = pd.DataFrame(index=df.index)

    if c_date:
        out["Date"] = pd.to_datetime(df[c_date], errors="coerce")
    else:
        out["Date"] = pd.to_datetime(range(len(df)), unit="D", origin="1970-01-01")

    if c_close is None:
        raise ValueError("Input data is missing a close-price column.")

    out["Close"] = pd.to_numeric(df[c_close], errors="coerce")
    out["Open"] = pd.to_numeric(df[c_open], errors="coerce") if c_open else out["Close"].copy()
    out["High"] = pd.to_numeric(df[c_high], errors="coerce") if c_high else out["Close"].copy()
    out["Low"] = pd.to_numeric(df[c_low], errors="coerce") if c_low else out["Close"].copy()

    if c_volume:
        out["Volume"] = pd.to_numeric(df[c_volume], errors="coerce")

    out["Open"] = out["Open"].fillna(out["Close"])
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out = out.sort_values("Date").reset_index(drop=True)

    return out


def get_price(row, which="close") -> float:
    which = str(which).lower()

    if which == "open":
        return float(row["Open"])
    if which == "close":
        return float(row["Close"])
    if which == "high":
        return float(row["High"])
    if which == "low":
        return float(row["Low"])

    raise ValueError("which must be one of: open, close, high, low.")


def calculate_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close)
    ])

    atr = pd.Series(tr).rolling(n, min_periods=1).mean()
    return pd.Series(atr.values, index=df.index)


def enforce_price_constraints(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray
):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    open_ = np.asarray(open_, dtype=np.float64)

    if open_.ndim == 2 and open_.shape[1] == 1:
        open_ = np.repeat(open_, close.shape[1], axis=1)

    high_corr = np.maximum(high, np.maximum(open_, close))
    low_corr = np.minimum(low, np.minimum(open_, close))

    return high_corr, low_corr, close


def inverse_prediction_block(*args, **kwargs):
    raise NotImplementedError(
        "Prediction inverse-transformation logic is omitted in the public version."
    )


def prepare_prediction_dataframe(*args, **kwargs):
    raise NotImplementedError(
        "Prediction-data alignment logic is omitted in the public version."
    )


def run_model_inference(*args, **kwargs):
    raise NotImplementedError(
        "Model inference logic is omitted in the public version."
    )


def apply_evaluation_period_filter(*args, **kwargs):
    raise NotImplementedError(
        "Evaluation-period filtering logic is omitted in the public version."
    )


def backtest_strategy_a(*args, **kwargs):
    raise NotImplementedError(
        "Strategy A trading logic is omitted in the public version."
    )


def backtest_strategy_b(*args, **kwargs):
    raise NotImplementedError(
        "Strategy B trading logic is omitted in the public version."
    )


def backtest_strategy_c(*args, **kwargs):
    raise NotImplementedError(
        "Strategy C trading logic is omitted in the public version."
    )


def equity_curve_from_trades(
    trades: pd.DataFrame,
    df: pd.DataFrame,
    initial_capital: float = 100000.0,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0
) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame({
            "Date": df["Date"],
            "equity": initial_capital
        })

    entry_map = {}
    exit_map = {}

    for _, row in trades.iterrows():
        entry_map[int(row.entry_idx)] = float(row.entry_price)
        exit_map[int(row.exit_idx)] = float(row.exit_price)

    capital = float(initial_capital)
    equity = np.full(len(df), np.nan)
    in_position = False
    shares = 0.0

    for i in range(len(df)):
        close_price = float(df.loc[i, "Close"])

        if i in entry_map and not in_position:
            entry_price = entry_map[i] * (1.0 + (fee_bps + slippage_bps) / 1e4)
            if entry_price > 0:
                shares = capital / entry_price
                capital -= shares * entry_price
                in_position = True

        if i in exit_map and in_position:
            exit_price = exit_map[i] * (1.0 - (fee_bps + slippage_bps) / 1e4)
            capital += shares * exit_price
            shares = 0.0
            in_position = False

        equity[i] = capital + shares * close_price if in_position else capital

    return pd.DataFrame({
        "Date": df["Date"],
        "equity": pd.Series(equity).ffill()
    })


def downside_std(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    returns = returns.dropna()

    if returns.empty:
        return 0.0

    downside = np.minimum(0.0, returns - risk_free_rate / 252.0)
    return float(np.sqrt((downside ** 2).mean()))


def calculate_performance_metrics(
    equity_df: pd.DataFrame,
    full_price_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252
) -> Dict[str, float]:
    equity_df = equity_df.dropna()

    if equity_df.empty or len(equity_df) < 2:
        return {
            "TotalReturn(%)": 0.0,
            "B&H(%)": 0.0,
            "MaxDrawdown(%)": 0.0,
            "Sharpe": 0.0,
            "CAGR(%)": 0.0,
            "Sortino": 0.0,
            "Calmar": 0.0,
        }

    init_eq = float(equity_df["equity"].iloc[0])
    final_eq = float(equity_df["equity"].iloc[-1])
    total_return = (final_eq / init_eq - 1.0) * 100.0

    start_date = equity_df["Date"].iloc[0]
    end_date = equity_df["Date"].iloc[-1]

    benchmark = full_price_df[
        (full_price_df["Date"] >= start_date) &
        (full_price_df["Date"] <= end_date)
    ]

    if benchmark.empty:
        bnh_return = 0.0
    else:
        bnh_return = (
            float(benchmark["Close"].iloc[-1]) /
            float(benchmark["Close"].iloc[0]) - 1.0
        ) * 100.0

    running_max = equity_df["equity"].cummax()
    max_drawdown = float(((equity_df["equity"] - running_max) / running_max).min() * 100.0)

    daily_returns = equity_df["equity"].pct_change().dropna()
    mean_return = float(daily_returns.mean()) if len(daily_returns) else 0.0
    std_return = float(daily_returns.std()) if len(daily_returns) > 1 else 0.0

    if std_return == 0 or np.isnan(std_return):
        sharpe = 0.0
    else:
        sharpe = (
            mean_return * trading_days_per_year - risk_free_rate
        ) / (std_return * np.sqrt(trading_days_per_year))

    years = max(
        1,
        int((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days)
    ) / 365.25

    cagr = (final_eq / init_eq) ** (1 / years) - 1.0

    dstd = downside_std(daily_returns, risk_free_rate)
    sortino = 0.0 if dstd == 0 else (
        mean_return * trading_days_per_year - risk_free_rate
    ) / (dstd * np.sqrt(trading_days_per_year))

    calmar = 0.0 if max_drawdown == 0 else cagr / (abs(max_drawdown) / 100.0)

    return {
        "TotalReturn(%)": round(total_return, 2),
        "B&H(%)": round(bnh_return, 2),
        "MaxDrawdown(%)": round(max_drawdown, 2),
        "Sharpe": round(sharpe, 2),
        "CAGR(%)": round(cagr * 100.0, 2),
        "Sortino": round(sortino, 2),
        "Calmar": round(calmar, 2),
    }


def summarize_trades(trades: pd.DataFrame) -> Dict[str, float]:
    empty = {
        "trades": 0,
        "winrate_%": 0.0,
        "avg_ret_%": 0.0,
        "median_ret_%": 0.0,
        "best_ret_%": 0.0,
        "worst_ret_%": 0.0,
        "cum_return_%": 0.0,
        "profit_factor": 0.0,
        "avg_hold_days": 0.0,
    }

    if trades is None or trades.empty:
        return empty

    returns = trades["ret_pct"].astype(float)
    total = len(trades)
    cum_mult = float(np.prod(1.0 + returns.values / 100.0))
    positive = returns[returns > 0].sum()
    negative = -returns[returns < 0].sum()
    profit_factor = float(positive / negative) if negative > 0 else 0.0

    return {
        "trades": total,
        "winrate_%": round(float((returns > 0).mean() * 100.0), 2),
        "avg_ret_%": round(float(returns.mean()), 3),
        "median_ret_%": round(float(returns.median()), 3),
        "best_ret_%": round(float(returns.max()), 3),
        "worst_ret_%": round(float(returns.min()), 3),
        "cum_return_%": round((cum_mult - 1.0) * 100.0, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_hold_days": round(float(trades["holding_days"].mean()), 1)
        if "holding_days" in trades.columns else 0.0,
    }


def collect_summary(
    run_dict: Dict[int, pd.DataFrame],
    mode_name: str,
    full_price_df: pd.DataFrame
) -> List[Dict]:
    rows = []

    for K, trades in run_dict.items():
        if trades is None or trades.empty:
            perf = {
                "TotalReturn(%)": 0.0,
                "B&H(%)": 0.0,
                "CAGR(%)": 0.0,
                "Sharpe": 0.0,
                "Sortino": 0.0,
                "Calmar": 0.0,
                "MaxDrawdown(%)": 0.0,
            }
            summ = {
                "profit_factor": 0.0,
                "winrate_%": 0.0,
                "trades": 0,
                "avg_hold_days": 0.0,
            }
        else:
            equity = equity_curve_from_trades(trades, full_price_df)
            perf = calculate_performance_metrics(equity, full_price_df)
            summ = summarize_trades(trades)

        rows.append({
            "Strategy": mode_name,
            "H": K,
            "TotalReturn(%)": perf["TotalReturn(%)"],
            "B&H(%)": perf["B&H(%)"],
            "CAGR(%)": perf["CAGR(%)"],
            "Sharpe": perf["Sharpe"],
            "Sortino": perf["Sortino"],
            "Calmar": perf["Calmar"],
            "MaxDrawdown(%)": perf["MaxDrawdown(%)"],
            "ProfitFactor": summ["profit_factor"],
            "WinRate(%)": summ["winrate_%"],
            "Trades": summ["trades"],
            "AvgHoldDays": summ["avg_hold_days"],
        })

    return rows


def plot_equity_curve(
    equity_df: pd.DataFrame,
    title: str,
    benchmark_df: Optional[pd.DataFrame] = None
):
    if equity_df is None or equity_df.empty:
        return

    dfp = equity_df.dropna()

    plt.figure(figsize=(12, 6))
    plt.plot(dfp["Date"], dfp["equity"], linewidth=1.8, label="Strategy")

    if benchmark_df is not None and not benchmark_df.empty:
        bench = benchmark_df.dropna().copy()
        bench["equity"] = dfp["equity"].iloc[0] * (
            bench["equity"] / bench["equity"].iloc[0]
        )
        plt.plot(
            bench["Date"],
            bench["equity"],
            linewidth=1.5,
            linestyle="--",
            alpha=0.8,
            label="Benchmark"
        )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_trades_on_price(
    df: pd.DataFrame,
    trades: pd.DataFrame,
    title: str
):
    if df is None or df.empty:
        return

    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["Close"], label="Close", linewidth=1.4)

    if trades is not None and not trades.empty:
        plt.scatter(
            trades["entry_date"],
            trades["entry_price"],
            marker="^",
            s=80,
            label="Entry"
        )
        plt.scatter(
            trades["exit_date"],
            trades["exit_price"],
            marker="v",
            s=80,
            label="Exit"
        )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_all_backtests(*args, **kwargs):
    raise NotImplementedError(
        "Full backtesting execution is omitted in the public version."
    )


def build_performance_table(*args, **kwargs):
    raise NotImplementedError(
        "Performance table generation is omitted in the public version."
    )


def build_attribution_table(*args, **kwargs):
    raise NotImplementedError(
        "Attribution analysis logic is omitted in the public version."
    )


def generate_final_report(*args, **kwargs):
    raise NotImplementedError(
        "Final report generation is omitted in the public version."
    )