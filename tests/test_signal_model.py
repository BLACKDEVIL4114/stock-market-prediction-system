from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from models.signal_model import SignalModel


class SignalModelTests(unittest.TestCase):
    def synthetic_data(self, rows: int = 220) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        close = 100 * (1 + pd.Series(rng.normal(0.001, 0.02, rows))).cumprod()
        high = close * (1 + rng.normal(0.01, 0.004, rows).clip(0.001, 0.03))
        low = close * (1 - rng.normal(0.01, 0.004, rows).clip(0.001, 0.03))
        open_ = close.shift(1).fillna(close.iloc[0])
        volume = pd.Series(rng.integers(1_000_000, 4_000_000, rows))
        return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})

    def test_feature_engineering_and_predict(self) -> None:
        model = SignalModel(config={"signals": {"lookback_window": 30, "target_horizon": 3, "min_confidence_threshold": 0.2}})
        df = self.synthetic_data()
        features = model.engineer_features(df)
        self.assertGreater(len(features.columns), 10)
        model.scaler.fit(features)
        model.model.predict = lambda X, verbose=0: np.array([[0.72, 0.18, 0.10]])
        prediction = model.predict(df)
        self.assertEqual(prediction["signal"], "BUY")
        self.assertGreaterEqual(prediction["confidence"], 0.7)


if __name__ == "__main__":
    unittest.main()
