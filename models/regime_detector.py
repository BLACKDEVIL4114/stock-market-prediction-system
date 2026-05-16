from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:  # pragma: no cover - fallback for environments without hmmlearn
    GaussianHMM = None


REGIME_LABELS = ["TRENDING_UP", "TRENDING_DOWN", "SIDEWAYS", "HIGH_VOLATILITY", "CRASH"]


@dataclass
class RegimeResult:
    regime: str
    confidence: float
    since: str


class RegimeDetector:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.hmm_model = self._build_hmm_model()
        self.rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
        self.state_to_label: dict[int, str] = {idx: label for idx, label in enumerate(REGIME_LABELS)}
        self.transition_matrix_: np.ndarray | None = None
        self._is_fitted = False

    @staticmethod
    def _build_hmm_model() -> Any:
        if GaussianHMM is not None:
            return GaussianHMM(n_components=5, covariance_type="diag", n_iter=200, random_state=42)
        return _FallbackHMM(n_components=5)

    @staticmethod
    def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        tr = pd.concat(
            [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(period).mean().replace(0, np.nan)
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
        return dx.rolling(period).mean()

    def prepare_features(self, df: pd.DataFrame, india_vix: pd.Series | None = None) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        features["return_5d"] = df["Close"].pct_change(5)
        features["vol_20d"] = df["Close"].pct_change().rolling(20).std() * np.sqrt(252)
        features["adx"] = self._adx(df)
        if india_vix is None:
            india_vix = features["vol_20d"].mul(100).bfill()
        features["india_vix"] = india_vix.reindex(df.index).ffill().bfill()
        return features.dropna()

    def _rule_based_label(self, row: pd.Series) -> str:
        if row["return_5d"] < -0.08 and row["vol_20d"] > 0.35:
            return "CRASH"
        if row["vol_20d"] > 0.30:
            return "HIGH_VOLATILITY"
        if row["adx"] > 25 and row["return_5d"] > 0:
            return "TRENDING_UP"
        if row["adx"] > 25 and row["return_5d"] <= 0:
            return "TRENDING_DOWN"
        return "SIDEWAYS"

    def _pseudo_labels(self, features: pd.DataFrame) -> pd.Series:
        return features.apply(self._rule_based_label, axis=1)

    def fit(self, df: pd.DataFrame, india_vix: pd.Series | None = None) -> None:
        features = self.prepare_features(df, india_vix=india_vix)
        if features.empty:
            raise ValueError("Not enough data to fit regime detector.")
        self.hmm_model.fit(features.values)
        hidden_states = self.hmm_model.predict(features.values)
        state_summary = pd.DataFrame({"state": hidden_states}, index=features.index).join(features[["return_5d", "vol_20d"]])
        for state in sorted(np.unique(hidden_states)):
            avg_return = state_summary.loc[state_summary["state"] == state, "return_5d"].mean()
            avg_vol = state_summary.loc[state_summary["state"] == state, "vol_20d"].mean()
            if avg_return < -0.08 and avg_vol > 0.30:
                label = "CRASH"
            elif avg_vol > 0.30:
                label = "HIGH_VOLATILITY"
            elif avg_return > 0:
                label = "TRENDING_UP"
            elif avg_return < 0:
                label = "TRENDING_DOWN"
            else:
                label = "SIDEWAYS"
            self.state_to_label[int(state)] = label
        y = self._pseudo_labels(features)
        self.rf_model.fit(features, y)
        self.transition_matrix_ = self.hmm_model.transmat_
        self._is_fitted = True

    def predict(self, df: pd.DataFrame | None = None, india_vix: pd.Series | None = None) -> dict[str, Any]:
        if df is None:
            return {"regime": "SIDEWAYS", "confidence": 0.5, "since": pd.Timestamp.utcnow().date().isoformat()}
        if not self._is_fitted:
            self.fit(df, india_vix=india_vix)
        features = self.prepare_features(df, india_vix=india_vix)
        latest = features.iloc[[-1]]
        hmm_state = int(self.hmm_model.predict(latest.values)[0])
        hmm_regime = self.state_to_label.get(hmm_state, "SIDEWAYS")
        rf_regime = str(self.rf_model.predict(latest)[0])
        rule_regime = self._rule_based_label(latest.iloc[0])
        votes = [hmm_regime, rf_regime, rule_regime]
        final_regime = max(set(votes), key=votes.count)
        rf_proba = 0.50
        if hasattr(self.rf_model, "predict_proba"):
            class_index = list(self.rf_model.classes_).index(rf_regime)
            rf_proba = float(self.rf_model.predict_proba(latest)[0][class_index])
        confidence = round((votes.count(final_regime) / 3) * 0.6 + rf_proba * 0.4, 2)
        latest_index = features.index[-1]
        since = latest_index.date().isoformat() if hasattr(latest_index, "date") else str(latest_index)
        return {"regime": final_regime, "confidence": confidence, "since": since}

    def get_strategy_params(self, regime: str) -> dict[str, Any]:
        params = {
            "TRENDING_UP": {"allow_long": True, "allow_short": False, "position_multiplier": 1.0, "stop_width": 1.2},
            "TRENDING_DOWN": {"allow_long": False, "allow_short": True, "position_multiplier": 0.5, "stop_width": 1.0},
            "SIDEWAYS": {"allow_long": True, "allow_short": True, "position_multiplier": 0.7, "stop_width": 0.8},
            "HIGH_VOLATILITY": {"allow_long": True, "allow_short": True, "position_multiplier": 0.5, "stop_width": 1.5},
            "CRASH": {"allow_long": False, "allow_short": False, "position_multiplier": 0.0, "stop_width": 0.0},
        }
        return params.get(regime, params["SIDEWAYS"])

    def transition_matrix(self) -> pd.DataFrame:
        if self.transition_matrix_ is None:
            raise ValueError("Regime detector has not been fitted yet.")
        return pd.DataFrame(self.transition_matrix_, columns=REGIME_LABELS, index=REGIME_LABELS)


class _FallbackHMM:
    def __init__(self, n_components: int = 5) -> None:
        self.n_components = n_components
        self.model = GaussianMixture(n_components=n_components, random_state=42)
        self.transmat_: np.ndarray = np.eye(n_components)
        self._states: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "_FallbackHMM":
        self.model.fit(X)
        self._states = self.model.predict(X)
        transition_counts = np.ones((self.n_components, self.n_components))
        for prev_state, next_state in zip(self._states[:-1], self._states[1:]):
            transition_counts[int(prev_state), int(next_state)] += 1
        self.transmat_ = transition_counts / transition_counts.sum(axis=1, keepdims=True)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
