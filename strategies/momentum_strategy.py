from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Any

class MomentumStrategy:
    symbol_map = {
        "NIFTY 50": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.lookback_days = self.config.get("market", {}).get("lookback_days", 365)
        self.interval = self.config.get("market", {}).get("bar_interval", "1d")

    def fetch_data(self, symbol: str) -> pd.DataFrame:
        ticker = self.symbol_map.get(symbol, symbol)
        df = yf.download(ticker, period=f"{self.lookback_days}d", interval=self.interval, auto_adjust=False, progress=False)
        if df.empty and not ticker.endswith(".NS"):
            df = yf.download(f"{ticker}.NS", period=f"{self.lookback_days}d", interval=self.interval, auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna() if not df.empty else pd.DataFrame()

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _supertrend(self, df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        atr = self._atr(df, period)
        hl2 = (df["High"] + df["Low"]) / 2
        final_upperband = hl2 + (multiplier * atr)
        final_lowerband = hl2 - (multiplier * atr)
        
        supertrend = pd.Series(True, index=df.index)
        for i in range(1, len(df.index)):
            if df["Close"].iloc[i] > final_upperband.iloc[i-1]:
                supertrend.iloc[i] = True
            elif df["Close"].iloc[i] < final_lowerband.iloc[i-1]:
                supertrend.iloc[i] = False
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                if supertrend.iloc[i] and final_lowerband.iloc[i] < final_lowerband.iloc[i-1]:
                    final_lowerband.iloc[i] = final_lowerband.iloc[i-1]
                if not supertrend.iloc[i] and final_upperband.iloc[i] > final_upperband.iloc[i-1]:
                    final_upperband.iloc[i] = final_upperband.iloc[i-1]
                    
        return pd.DataFrame({
            "Supertrend": supertrend,
            "Lowerband": final_lowerband,
            "Upperband": final_upperband
        })

    def analyze(self, df: pd.DataFrame) -> dict[str, Any]:
        """Runs advanced technical analysis on the dataframe to return a signal"""
        if len(df) < 50:
            return {"strategy_signal": "HOLD", "score": 50}
            
        st_df = self._supertrend(df)
        df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
        
        last_close = df["Close"].iloc[-1]
        last_ema20 = df["EMA_20"].iloc[-1]
        last_ema50 = df["EMA_50"].iloc[-1]
        last_st = st_df["Supertrend"].iloc[-1]
        
        score = 50
        if last_ema20 > last_ema50:
            score += 20
        else:
            score -= 20
            
        if last_st:
            score += 30
        else:
            score -= 30
            
        signal = "HOLD"
        if score >= 80:
            signal = "BUY"
        elif score <= 20:
            signal = "SELL"
            
        return {
            "strategy_signal": signal,
            "score": score,
            "trend_active": bool(last_st)
        }
