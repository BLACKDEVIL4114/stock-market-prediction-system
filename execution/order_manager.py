from __future__ import annotations

from typing import Any


class OrderManager:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.open_positions: list[dict[str, Any]] = []
        self.portfolio_state: dict[str, Any] = {
            "daily_pnl": 0.0,
            "drawdown_pct": 0.0,
            "consecutive_losses": 0,
            "returns": [],
            "market_circuit": False,
        }

    def place_order(self, symbol: str, signal: dict[str, Any], guardrail: dict[str, Any]) -> dict[str, Any]:
        order = {
            "symbol": symbol,
            "side": signal["signal"],
            "market_value": guardrail["adjusted_size"],
            "sector": symbol,
            "stop_loss": guardrail["stop_loss"],
        }
        self.open_positions.append(order)
        return order
