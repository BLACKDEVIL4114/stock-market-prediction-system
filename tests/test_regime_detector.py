from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from models.regime_detector import RegimeDetector


class RegimeDetectorTests(unittest.TestCase):
    @staticmethod
    def synthetic_data(rows: int = 300) -> pd.DataFrame:
        rng = np.random.default_rng(21)
        close = 200 * (1 + pd.Series(rng.normal(0.001, 0.02, rows))).cumprod()
        high = close * (1 + rng.normal(0.01, 0.005, rows).clip(0.001, 0.03))
        low = close * (1 - rng.normal(0.01, 0.005, rows).clip(0.001, 0.03))
        open_ = close.shift(1).fillna(close.iloc[0])
        volume = pd.Series(rng.integers(900_000, 4_000_000, rows))
        return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})

    def test_fit_and_predict(self) -> None:
        detector = RegimeDetector()
        df = self.synthetic_data()
        india_vix = pd.Series(np.linspace(12, 18, len(df)), index=df.index)
        detector.fit(df, india_vix=india_vix)
        result = detector.predict(df, india_vix=india_vix)
        self.assertIn(result["regime"], {"TRENDING_UP", "TRENDING_DOWN", "SIDEWAYS", "HIGH_VOLATILITY", "CRASH"})
        self.assertGreaterEqual(result["confidence"], 0)


if __name__ == "__main__":
    unittest.main()
