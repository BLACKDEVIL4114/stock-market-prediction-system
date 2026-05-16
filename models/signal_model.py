from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical


@dataclass
class SignalPrediction:
    signal: str
    confidence: float
    probabilities: dict[str, float]


class SignalModel:
    feature_names = [
        "rsi_14",
        "macd",
        "macd_signal",
        "bb_upper",
        "bb_lower",
        "ema_20",
        "ema_50",
        "vwap",
        "atr_14",
        "volume_ratio",
        "return_3d",
        "return_5d",
        "return_10d",
        "close_pct_change",
        "hl_range",
    ]

    def __init__(self, config: dict[str, Any] | None = None, model_dir: str | Path = "models") -> None:
        self.config = config or {}
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.lookback_window = self.config.get("signals", {}).get("lookback_window", 60)
        self.target_horizon = self.config.get("signals", {}).get("target_horizon", 5)
        self.min_confidence_threshold = self.config.get("signals", {}).get("min_confidence_threshold", 0.65)
        self.version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.scaler = StandardScaler()
        self.model = self._build_model()

    def _build_model(self) -> Sequential:
        model = Sequential(
            [
                LSTM(128, return_sequences=True, input_shape=(self.lookback_window, len(self.feature_names))),
                Dropout(0.2),
                LSTM(64),
                Dense(32, activation="relu"),
                Dense(3, activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = -delta.clip(upper=0).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        close = frame["Close"]
        volume = frame["Volume"].replace(0, np.nan)
        ema_20 = self._ema(close, 20)
        ema_50 = self._ema(close, 50)
        macd = self._ema(close, 12) - self._ema(close, 26)
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        rolling_mean = close.rolling(20).mean()
        rolling_std = close.rolling(20).std()
        cumulative_turnover = (close * volume).cumsum()
        cumulative_volume = volume.cumsum().replace(0, np.nan)

        features = pd.DataFrame(index=frame.index)
        features["rsi_14"] = self._rsi(close, 14)
        features["macd"] = macd
        features["macd_signal"] = macd_signal
        features["bb_upper"] = rolling_mean + 2 * rolling_std
        features["bb_lower"] = rolling_mean - 2 * rolling_std
        features["ema_20"] = ema_20
        features["ema_50"] = ema_50
        features["vwap"] = cumulative_turnover / cumulative_volume
        features["atr_14"] = self._atr(frame, 14)
        features["volume_ratio"] = volume / volume.rolling(20).mean()
        features["return_3d"] = close.pct_change(3)
        features["return_5d"] = close.pct_change(5)
        features["return_10d"] = close.pct_change(10)
        features["close_pct_change"] = close.pct_change()
        features["hl_range"] = (frame["High"] - frame["Low"]) / close.replace(0, np.nan)
        return features.dropna()

    def _create_targets(self, close: pd.Series) -> pd.Series:
        future_return = close.shift(-self.target_horizon) / close - 1
        labels = pd.Series(1, index=close.index)
        labels.loc[future_return > 0.02] = 0
        labels.loc[future_return < -0.02] = 2
        return labels

    def _build_sequences(self, features: pd.DataFrame, labels: pd.Series | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        scaled = self.scaler.fit_transform(features) if labels is not None else self.scaler.transform(features)
        X: list[np.ndarray] = []
        y: list[int] = []
        for idx in range(self.lookback_window, len(features)):
            X.append(scaled[idx - self.lookback_window : idx])
            if labels is not None:
                y.append(int(labels.iloc[idx]))
        X_array = np.asarray(X, dtype=np.float32)
        if labels is None:
            return X_array, None
        y_array = to_categorical(np.asarray(y, dtype=np.int32), num_classes=3)
        return X_array, y_array

    def train(self, df: pd.DataFrame, epochs: int = 3, batch_size: int = 16) -> dict[str, float]:
        features = self.engineer_features(df)
        labels = self._create_targets(df.loc[features.index, "Close"])
        X, y = self._build_sequences(features, labels)
        if len(X) == 0:
            raise ValueError("Not enough data to train the signal model.")
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.2)
        self.save()
        return {
            "loss": float(history.history["loss"][-1]),
            "accuracy": float(history.history["accuracy"][-1]),
        }

    def predict(self, df: pd.DataFrame) -> dict[str, Any]:
        features = self.engineer_features(df)
        if len(features) < self.lookback_window:
            return {"signal": "HOLD", "confidence": 0.0, "probabilities": {"BUY": 0.0, "HOLD": 1.0, "SELL": 0.0}}
        if not hasattr(self.scaler, "mean_"):
            self.scaler.fit(features)
        X, _ = self._build_sequences(features.tail(self.lookback_window + 1), None)
        probabilities = self.model.predict(X[-1:], verbose=0)[0]
        labels = ["BUY", "HOLD", "SELL"]
        best_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[best_idx])
        signal = labels[best_idx] if confidence >= self.min_confidence_threshold else "HOLD"
        return {
            "signal": signal,
            "confidence": confidence,
            "probabilities": {label: float(prob) for label, prob in zip(labels, probabilities)},
        }

    def save(self) -> dict[str, str]:
        versioned_model = self.model_dir / f"signal_model_{self.version}.keras"
        versioned_scaler = self.model_dir / f"signal_scaler_{self.version}.pkl"
        self.model.save(versioned_model)
        joblib.dump(self.scaler, versioned_scaler)
        joblib.dump(
            {"version": self.version, "model_path": str(versioned_model), "scaler_path": str(versioned_scaler)},
            self.model_dir / "signal_model_latest.pkl",
        )
        return {"model_path": str(versioned_model), "scaler_path": str(versioned_scaler)}

    def load(self, version: str | None = None) -> None:
        if version:
            model_path = self.model_dir / f"signal_model_{version}.keras"
            scaler_path = self.model_dir / f"signal_scaler_{version}.pkl"
        else:
            metadata = joblib.load(self.model_dir / "signal_model_latest.pkl")
            model_path = Path(metadata["model_path"])
            scaler_path = Path(metadata["scaler_path"])
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
