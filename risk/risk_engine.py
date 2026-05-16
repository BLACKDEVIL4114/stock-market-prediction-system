from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


class RiskEngine:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        risk = self.config.get("risk", {})
        self.capital = float(risk.get("capital", 100000))
        self.max_risk_per_trade_pct = float(risk.get("max_risk_per_trade_pct", 0.02))
        self.max_sector_exposure_pct = float(risk.get("max_sector_exposure_pct", 0.20))
        self.max_open_positions = int(risk.get("max_open_positions", 5))
        self.max_portfolio_var_pct = float(risk.get("max_portfolio_var_pct", 0.03))
        self.daily_loss_limit = float(risk.get("daily_loss_circuit_breaker", 3000))
        self.drawdown_limit_pct = float(risk.get("drawdown_circuit_breaker_pct", 0.05))
        self.consecutive_losses_limit = int(risk.get("consecutive_losses_limit", 3))
        self.risk_score_hard_stop = int(risk.get("risk_score_hard_stop", 80))

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> float:
        tr = pd.concat(
            [
                df["High"] - df["Low"],
                (df["High"] - df["Close"].shift()).abs(),
                (df["Low"] - df["Close"].shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])

    def calculate_position_size(self, entry_price: float, stop_loss: float, win_rate: float = 0.55, payoff_ratio: float = 1.5) -> float:
        risk_amount = self.capital * self.max_risk_per_trade_pct
        per_share_risk = max(entry_price - stop_loss, 0.01)
        kelly_fraction = max(win_rate - ((1 - win_rate) / max(payoff_ratio, 0.1)), 0.0)
        raw_size = (risk_amount / per_share_risk) * max(kelly_fraction, 0.25)
        return round(raw_size * entry_price, 2)

    def atr_stop_loss(self, entry_price: float, df: pd.DataFrame) -> float:
        return round(entry_price - (2 * self.atr(df)), 2)

    @staticmethod
    def trailing_stop(entry_price: float, current_price: float, target_price: float, current_stop: float) -> float:
        if current_price >= target_price:
            return round(max(current_stop, entry_price + (current_price - entry_price) * 0.5), 2)
        return round(current_stop, 2)

    @staticmethod
    def time_stop_exit(now: pd.Timestamp | None = None) -> bool:
        ts = now or pd.Timestamp.now(tz="Asia/Kolkata")
        return ts.hour > 15 or (ts.hour == 15 and ts.minute >= 15)

    def circuit_breaker(self, portfolio_state: dict[str, Any], risk_score: int, market_circuit: bool = False) -> tuple[bool, str]:
        daily_loss = float(portfolio_state.get("daily_pnl", 0))
        drawdown_pct = float(portfolio_state.get("drawdown_pct", 0))
        consecutive_losses = int(portfolio_state.get("consecutive_losses", 0))
        if daily_loss <= -self.daily_loss_limit:
            return True, f"Daily loss breached ₹{self.daily_loss_limit:.0f}"
        if drawdown_pct >= self.drawdown_limit_pct:
            return True, f"Drawdown breached {self.drawdown_limit_pct * 100:.1f}%"
        if consecutive_losses >= self.consecutive_losses_limit:
            return True, "Too many consecutive losses"
        if risk_score > self.risk_score_hard_stop:
            return True, "Risk score above hard stop"
        if market_circuit:
            return True, "NSE circuit breaker triggered"
        return False, "Circuit breaker clear"

    def portfolio_var(self, returns: pd.Series | list[float]) -> float:
        series = pd.Series(returns, dtype=float)
        if series.empty:
            return 0.0
        return abs(float(np.percentile(series.dropna(), 5))) * np.sqrt(1)

    def _sector_exposure(self, symbol: str, open_positions: Iterable[dict[str, Any]]) -> float:
        same_sector_value = sum(pos["market_value"] for pos in open_positions if pos.get("sector") == symbol)
        return same_sector_value / self.capital if self.capital else 0.0

    def evaluate_trade(
        self,
        symbol: str,
        signal: dict[str, Any],
        risk_result: dict[str, Any],
        market_data: pd.DataFrame,
        open_positions: list[dict[str, Any]],
        portfolio_state: dict[str, Any],
        regime: dict[str, Any],
    ) -> dict[str, Any]:
        entry_price = float(market_data["Close"].iloc[-1])
        stop_loss = self.atr_stop_loss(entry_price, market_data)
        normal_size = self.calculate_position_size(entry_price, stop_loss)
        adjusted_size = min(normal_size, float(risk_result["suggested_position_size"]))
        if len(open_positions) >= self.max_open_positions:
            return {"approved": False, "reason": "Max open positions reached", "adjusted_size": 0.0, "stop_loss": stop_loss}
        if self._sector_exposure(symbol, open_positions) >= self.max_sector_exposure_pct:
            return {"approved": False, "reason": "Sector exposure limit reached", "adjusted_size": 0.0, "stop_loss": stop_loss}
        halted, reason = self.circuit_breaker(portfolio_state, int(risk_result["risk_score"]), market_circuit=portfolio_state.get("market_circuit", False))
        if halted:
            return {"approved": False, "reason": reason, "adjusted_size": 0.0, "stop_loss": stop_loss}
        var_pct = self.portfolio_var(portfolio_state.get("returns", pd.Series(dtype=float)))
        if var_pct > self.max_portfolio_var_pct:
            return {"approved": False, "reason": "Portfolio VaR too high", "adjusted_size": 0.0, "stop_loss": stop_loss}
        if regime.get("regime") == "CRASH":
            return {"approved": False, "reason": "Crash regime: halt new trades", "adjusted_size": 0.0, "stop_loss": stop_loss}
        if signal.get("signal") == "HOLD":
            return {"approved": False, "reason": "No actionable signal", "adjusted_size": 0.0, "stop_loss": stop_loss}
        return {"approved": True, "reason": "Trade approved", "adjusted_size": round(adjusted_size, 2), "stop_loss": stop_loss}
