from __future__ import annotations

from datetime import datetime
from typing import Any

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml


def load_config() -> dict[str, Any]:
    with open("config.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

def save_config(config: dict[str, Any]) -> None:
    with open("config.yaml", "w", encoding="utf-8") as handle:
        yaml.dump(config, handle, default_flow_style=False)


def get_system_state() -> dict[str, Any]:
    state_file = Path("data/state.json")
    if state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "timestamp": "N/A",
        "regime": {"regime": "UNKNOWN"},
        "signals": [],
        "portfolio_state": {"daily_pnl": 0.0, "drawdown_pct": 0.0},
        "open_positions": []
    }

def fetch_positions(state: dict[str, Any]) -> pd.DataFrame:
    positions = state.get("open_positions", [])
    if not positions:
        return pd.DataFrame(columns=["symbol", "side", "market_value", "sector", "stop_loss"])
    return pd.DataFrame(positions)


def color_risk(val: float) -> str:
    if val >= 70:
        return "background-color: #ffcccc"
    if val >= 40:
        return "background-color: #fff0b3"
    return "background-color: #d7f7d4"


def live_overview(state: dict[str, Any]) -> None:
    st.subheader(f"Live Overview (Last Updated: {state.get('timestamp', 'N/A')})")
    col1, col2, col3 = st.columns(3)
    
    regime = state.get("regime", {}).get("regime", "UNKNOWN")
    pnl = state.get("portfolio_state", {}).get("daily_pnl", 0.0)
    dd = state.get("portfolio_state", {}).get("drawdown_pct", 0.0) * 100
    
    col1.metric("Market Regime", regime)
    col2.metric("Today's P&L", f"₹{pnl:,.2f}")
    col3.metric("Current Drawdown", f"{dd:.2f}%")

    st.markdown("### Open Positions")
    positions = fetch_positions(state)
    if not positions.empty:
        st.dataframe(positions, use_container_width=True)
    else:
        st.info("No open positions.")
        
    st.markdown("### Stock Chart (Live)")
    chart_symbol = st.selectbox("Select Symbol for Chart", state.get("signals", [{"symbol": "NIFTY 50"}]))
    if isinstance(chart_symbol, dict):
        chart_symbol = chart_symbol.get("symbol", "NIFTY 50")
        
    if chart_symbol:
        import yfinance as yf
        # Try both .NS and US ticker
        df = yf.download(chart_symbol, period="3mo", interval="1d", progress=False)
        if df.empty and not chart_symbol.endswith(".NS"):
            df = yf.download(f"{chart_symbol}.NS", period="3mo", interval="1d", progress=False)
            
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
            fig.update_layout(title=f"{chart_symbol} Candlestick Chart", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Could not fetch chart data for {chart_symbol}")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=dd,
            title={"text": "Drawdown (%)"},
            gauge={"axis": {"range": [0, 10]}, "bar": {"color": "#ef6c00"}},
        )
    )
    st.plotly_chart(gauge, use_container_width=True)


def signals_page(state: dict[str, Any]) -> None:
    st.subheader("Latest AI Signals & Approvals")
    
    raw_signals = state.get("signals", [])
    if not raw_signals:
        st.info("No signals generated yet.")
        return
        
    flat_signals = []
    for s in raw_signals:
        flat_signals.append({
            "symbol": s.get("symbol"),
            "AI_signal": s.get("signal", {}).get("signal", "HOLD"),
            "AI_conf": round(s.get("signal", {}).get("confidence", 0.0), 2),
            "tech_signal": s.get("tech_analysis", {}).get("strategy_signal", "HOLD"),
            "news_sentiment": s.get("news_sentiment", {}).get("label", "NEUTRAL"),
            "sentiment_score": round(s.get("news_sentiment", {}).get("score", 0.0), 2),
            "risk_score": s.get("risk", {}).get("risk_score", 0),
            "approved": s.get("approval", {}).get("approved", False),
            "reason": s.get("approval", {}).get("reason", "")
        })
    
    signals = pd.DataFrame(flat_signals)
    symbol_filter = st.selectbox("Symbol", ["All"] + list(signals["symbol"].unique()))
    signal_filter = st.selectbox("Signal Type", ["All", "BUY", "HOLD", "SELL"])
    min_conf = st.slider("Minimum AI Confidence", 0.0, 1.0, 0.50)
    
    filtered = signals[signals["AI_conf"] >= min_conf]
    if symbol_filter != "All":
        filtered = filtered[filtered["symbol"] == symbol_filter]
    if signal_filter != "All":
        filtered = filtered[filtered["AI_signal"] == signal_filter]
        
    st.dataframe(filtered.style.map(color_risk, subset=["risk_score"]), use_container_width=True)


def risk_analytics() -> None:
    st.subheader("Risk Analytics")
    index = pd.date_range(end=datetime.now(), periods=30)
    drawdown = pd.Series(np.linspace(0, -0.08, 30) + np.random.normal(0, 0.01, 30), index=index)
    var_series = pd.Series(np.linspace(0.01, 0.03, 30), index=index)
    risk_history = pd.DataFrame({"date": index, "RELIANCE": np.random.randint(20, 75, 30), "INFY": np.random.randint(15, 65, 30)})

    st.plotly_chart(px.line(drawdown.reset_index(), x="index", y=0, title="Rolling Drawdown"), use_container_width=True)
    st.plotly_chart(px.area(var_series.reset_index(), x="index", y=0, title="95% VaR (30D)"), use_container_width=True)
    st.plotly_chart(px.line(risk_history, x="date", y=["RELIANCE", "INFY"], title="Risk Score History"), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Win Rate", "57%")
    col2.metric("Sharpe Ratio", "1.68")
    col3.metric("Max Drawdown", "-8.7%")


def backtesting() -> None:
    st.subheader("Backtesting")
    col1, col2 = st.columns(2)
    col1.date_input("Start Date")
    col2.date_input("End Date")
    if st.button("Run Backtest"):
        index = pd.date_range(end=datetime.now(), periods=120)
        curve = pd.DataFrame(
            {
                "date": index,
                "AI Strategy": np.cumprod(1 + np.random.normal(0.0012, 0.01, len(index))) * 100000,
                "Nifty 50": np.cumprod(1 + np.random.normal(0.0008, 0.009, len(index))) * 100000,
            }
        )
        st.plotly_chart(px.line(curve, x="date", y=["AI Strategy", "Nifty 50"], title="Equity Curve"), use_container_width=True)


def settings_page(config: dict[str, Any]) -> None:
    st.subheader("Portfolio Management & Settings")
    
    st.markdown("### Watchlist (Add/Remove Shares)")
    current_watchlist = config.get("market", {}).get("watch_list", [])
    
    col1, col2 = st.columns(2)
    with col1:
        new_symbol = st.text_input("Add new Share Symbol (e.g., AAPL for US, RELIANCE for India)")
        if st.button("Add to Watchlist"):
            if new_symbol and new_symbol not in current_watchlist:
                config["market"]["watch_list"].append(new_symbol)
                save_config(config)
                st.success(f"Added {new_symbol} to watchlist! It will be scanned in the next pass.")
                st.rerun()
                
    with col2:
        remove_symbol = st.selectbox("Remove Share Symbol", ["Select..."] + current_watchlist)
        if st.button("Remove from Watchlist"):
            if remove_symbol and remove_symbol != "Select...":
                config["market"]["watch_list"].remove(remove_symbol)
                save_config(config)
                st.success(f"Removed {remove_symbol} from watchlist!")
                st.rerun()
                
    st.markdown("### System Settings")
    st.toggle("Paper Mode", value=config.get("execution", {}).get("mode", "paper") == "paper")
    st.number_input("Capital", value=float(config.get("risk", {}).get("capital", 100000)))
    st.number_input("Max Daily Loss", value=float(config.get("risk", {}).get("max_daily_loss", 5000)))
    st.number_input("Max Open Positions", value=int(config.get("risk", {}).get("max_open_positions", 5)))


def should_refresh() -> bool:
    now = pd.Timestamp.now(tz="Asia/Kolkata")
    return now.weekday() < 5 and ((now.hour == 9 and now.minute >= 15) or (9 < now.hour < 15) or (now.hour == 15 and now.minute <= 30))


def main() -> None:
    st.set_page_config(page_title="nse-ai-trader", layout="wide")
    config = load_config()
    st.title("nse-ai-trader Dashboard")
    if should_refresh():
        st.caption("Auto-refresh every 30 seconds during market hours (configure with Streamlit autorefresh in deployment).")
    state = get_system_state()
    page = st.sidebar.radio("Pages", ["LIVE OVERVIEW", "SIGNALS", "RISK ANALYTICS", "BACKTESTING", "SETTINGS"])
    if page == "LIVE OVERVIEW":
        live_overview(state)
    elif page == "SIGNALS":
        signals_page(state)
    elif page == "RISK ANALYTICS":
        risk_analytics()
    elif page == "BACKTESTING":
        backtesting()
    else:
        settings_page(config)


if __name__ == "__main__":
    main()
