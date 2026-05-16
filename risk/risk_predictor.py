from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


@dataclass
class RiskPrediction:
    risk_score: int
    risk_level: str
    allow_trade: bool
    reasons: list[str]
    suggested_position_size: float
    normal_position_size: float


class RiskPredictor:
    weights = {
        "volatility_risk": 0.20,
        "drawdown_risk": 0.15,
        "liquidity_risk": 0.15,
        "regime_risk": 0.20,
        "correlation_risk": 0.10,
        "news_risk": 0.10,
        "time_risk": 0.10,
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.capital = float(self.config.get("risk", {}).get("capital", 100000))
        self.max_risk_per_trade = float(self.config.get("risk", {}).get("max_risk_per_trade_pct", 0.02))
        self.drawdown_model = XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
        self._is_fitted = False

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat(
            [
                df["High"] - df["Low"],
                (df["High"] - df["Close"].shift()).abs(),
                (df["Low"] - df["Close"].shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period).mean()

    def _bollinger_width(self, close: pd.Series, period: int = 20) -> pd.Series:
        mean = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = mean + 2 * std
        lower = mean - 2 * std
        return (upper - lower) / mean.replace(0, np.nan)

    def _prepare_drawdown_features(self, df: pd.DataFrame) -> pd.DataFrame:
        returns = df["Close"].pct_change().fillna(0)
        frame = pd.DataFrame(index=df.index)
        frame["ret_1d"] = returns
        frame["ret_5d"] = df["Close"].pct_change(5)
        frame["vol_20d"] = returns.rolling(20).std()
        frame["atr_pct"] = self._atr(df) / df["Close"]
        frame["volume_zscore"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()
        return frame.dropna()

    def train(self, df: pd.DataFrame) -> dict[str, float]:
        features = self._prepare_drawdown_features(df)
        rolling_peak = df["Close"].cummax()
        max_drawdown = ((df["Close"] - rolling_peak) / rolling_peak).abs()
        target = max_drawdown.reindex(features.index).ffill()
        self.drawdown_model.fit(features, target)
        self._is_fitted = True
        prediction = self.drawdown_model.predict(features)
        return {"r2_score": float(r2_score(target, prediction))}

    def _volatility_risk(self, df: pd.DataFrame) -> tuple[float, list[str]]:
        atr_pct = float((self._atr(df).iloc[-1] / df["Close"].iloc[-1]) * 100)
        bb_width = float(self._bollinger_width(df["Close"]).iloc[-1] * 100)
        score = min(100.0, atr_pct * 20 + bb_width * 3)
        reasons = []
        if atr_pct > 2:
            reasons.append(f"High volatility (ATR {atr_pct:.1f}%)")
        if bb_width > 12:
            reasons.append(f"Wide Bollinger bands ({bb_width:.1f}%)")
        return score, reasons

    def _drawdown_risk(self, df: pd.DataFrame) -> tuple[float, list[str]]:
        if not self._is_fitted:
            self.train(df)
        features = self._prepare_drawdown_features(df).tail(1)
        predicted_drawdown = float(self.drawdown_model.predict(features)[0] * 100)
        score = float(np.clip(predicted_drawdown * 5, 0, 100))
        reasons = [f"Predicted drawdown {predicted_drawdown:.1f}%"] if predicted_drawdown > 5 else []
        return score, reasons

    def _liquidity_risk(self, df: pd.DataFrame, bid_ask_spread: float | None = None) -> tuple[float, list[str]]:
        spread = bid_ask_spread if bid_ask_spread is not None else float(df["Close"].iloc[-1] * 0.0015)
        spread_pct = (spread / df["Close"].iloc[-1]) * 100
        volume_ratio = float(df["Volume"].iloc[-1] / df["Volume"].rolling(20).mean().iloc[-1])
        score = np.clip(spread_pct * 120 - volume_ratio * 15 + 40, 0, 100)
        reasons = []
        if volume_ratio < 0.7:
            reasons.append(f"Low liquidity (volume ratio {volume_ratio:.2f})")
        if spread_pct > 0.15:
            reasons.append(f"Wide spread ({spread_pct:.2f}%)")
        return float(score), reasons

    def _regime_risk(self, regime: str | None) -> tuple[float, list[str]]:
        mapping = {
            "TRENDING_UP": 20,
            "SIDEWAYS": 40,
            "TRENDING_DOWN": 65,
            "HIGH_VOLATILITY": 80,
            "CRASH": 100,
        }
        score = float(mapping.get(regime or "SIDEWAYS", 50))
        reasons = [f"Market regime {regime}"] if score >= 65 else []
        return score, reasons

    def _correlation_risk(self, symbol: str, open_positions: Iterable[dict[str, Any]] | None = None) -> tuple[float, list[str]]:
        open_positions = list(open_positions or [])
        same_sector_count = sum(1 for pos in open_positions if pos.get("sector") == symbol)
        score = float(min(100, same_sector_count * 25 + len(open_positions) * 8))
        reasons = ["Trade correlated with existing book"] if score > 50 else []
        return score, reasons

    def _news_risk(self, sentiment_score: float | None = None) -> tuple[float, list[str]]:
        if sentiment_score is None:
            sentiment_score = 0.0
        score = float(np.clip((0.5 - sentiment_score) * 100, 0, 100))
        reasons = [f"Negative news sentiment ({sentiment_score:.1f})"] if sentiment_score < -0.3 else []
        return score, reasons

    def _time_risk(self, timestamp: pd.Timestamp | None = None) -> tuple[float, list[str]]:
        ts = timestamp or pd.Timestamp.now(tz="Asia/Kolkata")
        minutes = ts.hour * 60 + ts.minute
        open_risk = 75 if 555 <= minutes <= 570 else 0
        close_risk = 75 if 900 <= minutes <= 915 else 0
        score = float(max(open_risk, close_risk))
        reasons = ["Trade near market open/close"] if score else []
        return score, reasons

    def predict(
        self,
        symbol: str,
        signal: dict[str, Any],
        df: pd.DataFrame | None = None,
        regime: str | None = None,
        open_positions: Iterable[dict[str, Any]] | None = None,
        sentiment_score: float | None = None,
        bid_ask_spread: float | None = None,
        timestamp: pd.Timestamp | None = None,
    ) -> dict[str, Any]:
        if df is None:
            df = self._synthetic_market_frame()
        volatility, reasons_v = self._volatility_risk(df)
        drawdown, reasons_d = self._drawdown_risk(df)
        liquidity, reasons_l = self._liquidity_risk(df, bid_ask_spread=bid_ask_spread)
        regime_score, reasons_r = self._regime_risk(regime)
        correlation, reasons_c = self._correlation_risk(symbol, open_positions)
        news, reasons_n = self._news_risk(sentiment_score)
        time_score, reasons_t = self._time_risk(timestamp)

        factor_scores = {
            "volatility_risk": volatility,
            "drawdown_risk": drawdown,
            "liquidity_risk": liquidity,
            "regime_risk": regime_score,
            "correlation_risk": correlation,
            "news_risk": news,
            "time_risk": time_score,
        }
        weighted_avg = sum(factor_scores[name] * weight for name, weight in self.weights.items())
        regime_multiplier = 1.15 if regime in {"HIGH_VOLATILITY", "CRASH"} else 1.0
        risk_score = int(np.clip(weighted_avg * regime_multiplier, 0, 100))
        normal_position_size = self.capital * self.max_risk_per_trade * max(signal.get("confidence", 0.5), 0.5)
        suggested_position_size = 0.0 if risk_score >= 80 else round(normal_position_size * max(0.1, 1 - risk_score / 100), 2)
        risk_level = "LOW" if risk_score < 35 else "MEDIUM" if risk_score < 65 else "HIGH"
        reasons = reasons_v + reasons_d + reasons_l + reasons_r + reasons_c + reasons_n + reasons_t
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "allow_trade": risk_score < 75,
            "reasons": reasons or ["No significant risk flags"],
            "suggested_position_size": suggested_position_size,
            "normal_position_size": round(normal_position_size, 2),
            "factor_scores": factor_scores,
        }

    def backtest_risk_accuracy(self, df: pd.DataFrame) -> dict[str, float]:
        if len(df) < 80:
            raise ValueError("Need at least 80 rows to backtest risk accuracy.")
        scores: list[int] = []
        realized_drawdowns: list[float] = []
        for idx in range(60, len(df) - 5):
            history = df.iloc[: idx + 1]
            score = self.predict("BACKTEST", {"confidence": 0.75}, df=history)["risk_score"]
            forward_window = df["Close"].iloc[idx : idx + 5]
            realized_dd = abs((forward_window.min() - forward_window.iloc[0]) / forward_window.iloc[0]) * 100
            scores.append(score)
            realized_drawdowns.append(realized_dd)
        correlation = float(np.corrcoef(scores, realized_drawdowns)[0, 1])
        if np.isnan(correlation):
            correlation = 0.0
        mae = float(np.mean(np.abs(np.asarray(scores) / 100 * 10 - np.asarray(realized_drawdowns))))
        return {"score_drawdown_correlation": correlation, "drawdown_mae": mae}

    @staticmethod
    def _synthetic_market_frame(rows: int = 240) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        close = 100 * (1 + pd.Series(rng.normal(0.001, 0.02, rows))).cumprod()
        high = close * (1 + rng.normal(0.01, 0.005, rows).clip(0.001, 0.03))
        low = close * (1 - rng.normal(0.01, 0.005, rows).clip(0.001, 0.03))
        open_ = close.shift(1).fillna(close.iloc[0])
        volume = pd.Series(rng.integers(1_000_000, 4_000_000, rows))
        return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})
