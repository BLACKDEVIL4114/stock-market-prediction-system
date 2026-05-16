from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from risk.risk_predictor import RiskPredictor


class RiskPredictorTests(unittest.TestCase):
    @staticmethod
    def synthetic_data(rows: int = 260) -> pd.DataFrame:
        rng = np.random.default_rng(11)
        close = 150 * (1 + pd.Series(rng.normal(0.001, 0.018, rows))).cumprod()
        high = close * (1 + rng.normal(0.012, 0.004, rows).clip(0.001, 0.03))
        low = close * (1 - rng.normal(0.012, 0.004, rows).clip(0.001, 0.03))
        open_ = close.shift(1).fillna(close.iloc[0])
        volume = pd.Series(rng.integers(800_000, 5_000_000, rows))
        return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})

    def test_predict_returns_expected_shape(self) -> None:
        predictor = RiskPredictor(config={"risk": {"capital": 100000, "max_risk_per_trade_pct": 0.02}})
        df = self.synthetic_data()
        predictor.train(df)
        result = predictor.predict("RELIANCE", {"confidence": 0.84}, df=df, regime="TRENDING_UP", sentiment_score=-0.6)
        self.assertIn("risk_score", result)
        self.assertIn(result["risk_level"], {"LOW", "MEDIUM", "HIGH"})
        self.assertIsInstance(result["reasons"], list)

    def test_backtest_accuracy_metrics(self) -> None:
        predictor = RiskPredictor()
        metrics = predictor.backtest_risk_accuracy(self.synthetic_data(320))
        self.assertIn("score_drawdown_correlation", metrics)
        self.assertIn("drawdown_mae", metrics)


if __name__ == "__main__":
    unittest.main()
