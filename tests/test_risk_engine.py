from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from risk.risk_engine import RiskEngine


class RiskEngineTests(unittest.TestCase):
    @staticmethod
    def synthetic_data(rows: int = 120) -> pd.DataFrame:
        rng = np.random.default_rng(31)
        close = 100 * (1 + pd.Series(rng.normal(0.001, 0.015, rows))).cumprod()
        high = close * (1 + rng.normal(0.01, 0.004, rows).clip(0.001, 0.03))
        low = close * (1 - rng.normal(0.01, 0.004, rows).clip(0.001, 0.03))
        open_ = close.shift(1).fillna(close.iloc[0])
        volume = pd.Series(rng.integers(800_000, 4_000_000, rows))
        return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})

    def setUp(self) -> None:
        self.engine = RiskEngine(config={"risk": {"capital": 100000, "risk_score_hard_stop": 80}})
        self.df = self.synthetic_data()

    def test_position_sizing(self) -> None:
        stop_loss = self.engine.atr_stop_loss(float(self.df["Close"].iloc[-1]), self.df)
        size = self.engine.calculate_position_size(float(self.df["Close"].iloc[-1]), stop_loss)
        self.assertGreater(size, 0)

    def test_circuit_breaker_rejects_high_daily_loss(self) -> None:
        halted, reason = self.engine.circuit_breaker({"daily_pnl": -3500, "drawdown_pct": 0.01, "consecutive_losses": 0}, 20)
        self.assertTrue(halted)
        self.assertIn("Daily loss", reason)

    def test_circuit_breaker_rejects_high_drawdown(self) -> None:
        halted, reason = self.engine.circuit_breaker({"daily_pnl": 0, "drawdown_pct": 0.06, "consecutive_losses": 0}, 20)
        self.assertTrue(halted)
        self.assertIn("Drawdown", reason)

    def test_circuit_breaker_rejects_consecutive_losses(self) -> None:
        halted, reason = self.engine.circuit_breaker({"daily_pnl": 0, "drawdown_pct": 0.01, "consecutive_losses": 3}, 20)
        self.assertTrue(halted)
        self.assertIn("consecutive", reason.lower())

    def test_circuit_breaker_rejects_high_risk_score(self) -> None:
        halted, reason = self.engine.circuit_breaker({"daily_pnl": 0, "drawdown_pct": 0.01, "consecutive_losses": 0}, 81)
        self.assertTrue(halted)
        self.assertIn("Risk score", reason)

    def test_circuit_breaker_rejects_market_halt(self) -> None:
        halted, reason = self.engine.circuit_breaker({"daily_pnl": 0, "drawdown_pct": 0.01, "consecutive_losses": 0}, 20, market_circuit=True)
        self.assertTrue(halted)
        self.assertIn("circuit breaker", reason.lower())

    def test_time_stop_exit(self) -> None:
        should_exit = self.engine.time_stop_exit(pd.Timestamp("2026-05-14 15:16:00", tz="Asia/Kolkata"))
        self.assertTrue(should_exit)

    def test_max_open_positions_rejects_trade(self) -> None:
        result = self.engine.evaluate_trade(
            symbol="RELIANCE",
            signal={"signal": "BUY", "confidence": 0.8},
            risk_result={"risk_score": 32, "suggested_position_size": 9000},
            market_data=self.df,
            open_positions=[{"market_value": 1000, "sector": "X"}] * 5,
            portfolio_state={"daily_pnl": 0, "drawdown_pct": 0.0, "consecutive_losses": 0, "returns": [0.01, -0.01]},
            regime={"regime": "TRENDING_UP"},
        )
        self.assertFalse(result["approved"])
        self.assertIn("Max open positions", result["reason"])

    def test_sector_exposure_rejects_trade(self) -> None:
        result = self.engine.evaluate_trade(
            symbol="RELIANCE",
            signal={"signal": "BUY", "confidence": 0.8},
            risk_result={"risk_score": 32, "suggested_position_size": 9000},
            market_data=self.df,
            open_positions=[{"market_value": 25000, "sector": "RELIANCE"}],
            portfolio_state={"daily_pnl": 0, "drawdown_pct": 0.0, "consecutive_losses": 0, "returns": [0.01, -0.01]},
            regime={"regime": "TRENDING_UP"},
        )
        self.assertFalse(result["approved"])
        self.assertIn("Sector exposure", result["reason"])

    def test_var_limit_rejects_trade(self) -> None:
        result = self.engine.evaluate_trade(
            symbol="RELIANCE",
            signal={"signal": "BUY", "confidence": 0.8},
            risk_result={"risk_score": 32, "suggested_position_size": 9000},
            market_data=self.df,
            open_positions=[],
            portfolio_state={"daily_pnl": 0, "drawdown_pct": 0.0, "consecutive_losses": 0, "returns": [-0.05, -0.04, -0.03]},
            regime={"regime": "TRENDING_UP"},
        )
        self.assertFalse(result["approved"])
        self.assertIn("VaR", result["reason"])

    def test_evaluate_trade(self) -> None:
        result = self.engine.evaluate_trade(
            symbol="RELIANCE",
            signal={"signal": "BUY", "confidence": 0.8},
            risk_result={"risk_score": 32, "suggested_position_size": 9000},
            market_data=self.df,
            open_positions=[],
            portfolio_state={"daily_pnl": 0, "drawdown_pct": 0.0, "consecutive_losses": 0, "returns": pd.Series([0.01, -0.01])},
            regime={"regime": "TRENDING_UP"},
        )
        self.assertTrue(result["approved"])


if __name__ == "__main__":
    unittest.main()
