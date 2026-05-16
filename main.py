from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from execution.order_manager import OrderManager
from models.regime_detector import RegimeDetector
from models.signal_model import SignalModel
from models.news_sentiment import NewsSentimentAnalyzer
from notifications.telegram_bot import TelegramNotifier
from risk.risk_engine import RiskEngine
from risk.risk_predictor import RiskPredictor
from strategies.momentum_strategy import MomentumStrategy


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run() -> dict[str, Any]:
    config = load_config()
    order_manager = OrderManager(config)
    signal_model = SignalModel(config=config)
    regime_detector = RegimeDetector(config=config)
    risk_predictor = RiskPredictor(config=config)
    risk_engine = RiskEngine(config=config)
    notifier = TelegramNotifier(config=config)
    strategy = MomentumStrategy(config=config)
    news_analyzer = NewsSentimentAnalyzer()

    watch_list = config["market"]["watch_list"]
    regime_snapshot = regime_detector.predict()
    outputs: dict[str, Any] = {"regime": regime_snapshot, "signals": []}

    for symbol in watch_list:
        market_frame = strategy.fetch_data(symbol)
        if market_frame.empty:
            continue

        signal = signal_model.predict(market_frame)
        tech_analysis = strategy.analyze(market_frame)
        news_sentiment = news_analyzer.analyze_symbol(symbol)
        
        # Confluence: Adjust AI confidence based on technical analysis & news sentiment
        if signal["signal"] == tech_analysis["strategy_signal"] and signal["signal"] != "HOLD":
            signal["confidence"] = min(1.0, signal["confidence"] + 0.15)
        elif tech_analysis["strategy_signal"] != "HOLD" and signal["signal"] != "HOLD" and signal["signal"] != tech_analysis["strategy_signal"]:
            signal["confidence"] = max(0.0, signal["confidence"] - 0.20)
            
        # Sentiment adjustment
        if news_sentiment["label"] == "POSITIVE" and signal["signal"] == "BUY":
            signal["confidence"] = min(1.0, signal["confidence"] + 0.10)
        elif news_sentiment["label"] == "NEGATIVE" and signal["signal"] == "SELL":
            signal["confidence"] = min(1.0, signal["confidence"] + 0.10)
        elif news_sentiment["label"] == "NEGATIVE" and signal["signal"] == "BUY":
            signal["confidence"] = max(0.0, signal["confidence"] - 0.25)
            
        if signal["confidence"] < config["signals"]["min_confidence_threshold"]:
            signal["signal"] = "HOLD"

        risk_result = risk_predictor.predict(symbol=symbol, signal=signal)
        guardrail = risk_engine.evaluate_trade(
            symbol=symbol,
            signal=signal,
            risk_result=risk_result,
            market_data=market_frame,
            open_positions=order_manager.open_positions,
            portfolio_state=order_manager.portfolio_state,
            regime=regime_snapshot,
        )
        payload = {
            "symbol": symbol,
            "signal": signal,
            "tech_analysis": tech_analysis,
            "news_sentiment": news_sentiment,
            "risk": risk_result,
            "approval": guardrail,
        }
        outputs["signals"].append(payload)
        notifier.queue_signal(payload, regime_snapshot)

        if guardrail["approved"] and signal["signal"] != "HOLD":
            order_manager.place_order(symbol, signal, guardrail)

    notifier.flush_queue()
    
    # Save state for dashboard safely
    import os
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)
            
    state = {
        "timestamp": datetime.now().isoformat(),
        "regime": regime_snapshot,
        "signals": outputs["signals"],
        "portfolio_state": order_manager.portfolio_state,
        "open_positions": order_manager.open_positions
    }
    Path("data").mkdir(exist_ok=True)
    temp_file = "data/state.json.tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4, cls=NumpyEncoder)
    os.replace(temp_file, "data/state.json")
        
    return outputs


if __name__ == "__main__":
    from datetime import datetime
    while True:
        print(f"[{datetime.now()}] Running AI Trader Pass...")
        try:
            result = run()
            print(f"[{datetime.now()}] Pass Complete. Saved State.")
        except Exception as e:
            print(f"[{datetime.now()}] Error: {e}")
        # Run every 5 minutes
        time.sleep(300)
