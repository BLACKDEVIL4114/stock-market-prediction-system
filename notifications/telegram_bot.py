from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes


@dataclass
class NotificationMessage:
    kind: str
    text: str


class TelegramNotifier:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        telegram_config = self.config.get("telegram", {})
        self.enabled = bool(telegram_config.get("enabled", False))
        self.bot_token = telegram_config.get("bot_token", "")
        self.chat_id = telegram_config.get("chat_id", "")
        self.paused = False
        self.queue: list[NotificationMessage] = []

    def queue_signal(self, payload: dict[str, Any], regime: dict[str, Any]) -> None:
        signal = payload["signal"]
        risk = payload["risk"]
        approval = payload["approval"]
        symbol = payload["symbol"]
        if signal["signal"] == "HOLD":
            return
        text = (
            f"🟢 {signal['signal']} signal: {symbol}\n"
            f"Confidence: {signal['confidence']:.0%} | Risk score: {risk['risk_score']} ({risk['risk_level']})\n"
            f"Entry: market | SL: ₹{approval['stop_loss']:.2f} | Target: discretionary\n"
            f"Regime: {regime['regime']} | Pos size: ₹{approval['adjusted_size']:.0f}\n"
            "Reply /approve or /reject"
        )
        self.queue.append(NotificationMessage(kind="signal", text=text))

    def queue_risk_alert(self, symbol: str, risk_score: int, reason: str, reduced_size: float) -> None:
        self.queue.append(
            NotificationMessage(
                kind="risk",
                text=(
                    f"🔴 HIGH RISK ALERT: {symbol}\n"
                    f"Risk score jumped to {risk_score} (HIGH)\n"
                    f"Reason: {reason}\n"
                    f"Action: Position size reduced to ₹{reduced_size:,.0f}"
                ),
            )
        )

    def queue_circuit_breaker(self, daily_loss: float, limit: float) -> None:
        self.queue.append(
            NotificationMessage(
                kind="circuit",
                text=(
                    "⚡ CIRCUIT BREAKER TRIGGERED\n"
                    f"Daily loss: ₹{daily_loss:,.0f} (limit: ₹{limit:,.0f})\n"
                    "All new trades HALTED for today\n"
                    "Open positions: protected with SL"
                ),
            )
        )

    def queue_daily_summary(self, summary: dict[str, Any]) -> None:
        self.queue.append(
            NotificationMessage(
                kind="summary",
                text=(
                    f"📊 Daily Summary - {summary['date']}\n"
                    f"Trades: {summary['trades']} | Win: {summary['wins']} | Loss: {summary['losses']}\n"
                    f"P&L: {summary['pnl']} ({summary['pnl_pct']})\n"
                    f"Best: {summary['best']} | Worst: {summary['worst']}\n"
                    f"Risk score avg: {summary['avg_risk']} | Regime: {summary['regime']}"
                ),
            )
        )

    def flush_queue(self) -> None:
        if not self.enabled:
            return
        for message in self.queue:
            print(f"[Telegram queued] {message.kind}: {message.text}")
        self.queue.clear()

    async def start(self) -> None:
        if not self.enabled or not self.bot_token:
            return
        app = Application.builder().token(self.bot_token).build()
        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("status", self._cmd_status))
        app.add_handler(CommandHandler("positions", self._cmd_positions))
        app.add_handler(CommandHandler("pause", self._cmd_pause))
        app.add_handler(CommandHandler("resume", self._cmd_resume))
        await app.initialize()
        await app.start()
        await app.updater.start_polling()

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("nse-ai-trader is online. Use /status, /positions, /pause, /resume.")

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        status = "paused" if self.paused else "running"
        await update.message.reply_text(f"System status: {status}")

    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("Open positions snapshot is connected at runtime by the orchestrator.")

    async def _cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.paused = True
        await update.message.reply_text("Trading paused.")

    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.paused = False
        await update.message.reply_text("Trading resumed.")


async def run_notifier(config: dict[str, Any]) -> None:
    notifier = TelegramNotifier(config=config)
    await notifier.start()


if __name__ == "__main__":
    asyncio.run(run_notifier({"telegram": {"enabled": False}}))
