# nse-ai-trader

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Stars](https://img.shields.io/github/stars/your-username/nse-ai-trader?style=social)](https://github.com/your-username/nse-ai-trader)

**An open-source AI algorithmic trading system for Indian markets (NSE/BSE)** combining signal generation, market regime detection, automated risk prediction, execution guardrails, and a live Streamlit control center.

## Why it stands out

- 🧠 **AI signal generation** with an LSTM model on engineered OHLCV features
- 🛡️ **Automated risk prediction** with multi-factor pre-trade scoring
- 🌦️ **Regime detection** using HMM + Random Forest + rule-based voting
- ⚡ **Circuit breakers** to protect capital during drawdowns and stressed conditions
- 📈 **Streamlit dashboard** for live overview, analytics, and backtesting
- 📲 **Telegram alerts** for signals, pauses, positions, and end-of-day summaries

## Architecture

```text
                    +----------------------+
                    |    Market Data       |
                    | yfinance / NSE feeds |
                    +----------+-----------+
                               |
                               v
 +-------------------+   +------------+   +--------------------+
 | SignalModel       |-->| Main Loop  |<--| RegimeDetector     |
 | LSTM classifier   |   | orchestrator|   | HMM + RF + rules   |
 +-------------------+   +------+-----+   +--------------------+
                                |
                                v
                      +--------------------+
                      | RiskPredictor      |
                      | 7-factor AI score  |
                      +---------+----------+
                                |
                                v
                      +--------------------+
                      | RiskEngine         |
                      | sizing + stops +   |
                      | circuit breakers   |
                      +-----+--------------+
                            |
               +------------+-------------+
               |                          |
               v                          v
      +-------------------+      +---------------------+
      | Execution Layer   |      | Dashboard / Alerts  |
      | Kite / paper mode |      | Streamlit / Telegram|
      +-------------------+      +---------------------+
```

## Quick start

```bash
git clone https://github.com/your-username/nse-ai-trader.git
cd nse-ai-trader
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then:

```bash
python main.py
streamlit run dashboard/app.py
python -m unittest discover -s tests
```

## Configuration

All runtime settings live in `config.yaml`.

| Key | Description | Example |
|---|---|---|
| `app.environment` | `paper` or `live` trading mode | `paper` |
| `market.benchmark` | Benchmark used in backtests and relative stats | `^NSEI` |
| `market.volatility_index_symbol` | India VIX proxy symbol | `^INDIAVIX` |
| `market.watch_list` | Symbols the system scans | `RELIANCE`, `INFY` |
| `market.bar_interval` | Candle interval for model inputs | `1d` |
| `market.lookback_days` | Historical lookback for training and risk | `365` |
| `risk.capital` | Starting portfolio capital | `100000` |
| `risk.max_daily_loss` | Max loss allowed per day | `5000` |
| `risk.max_risk_per_trade_pct` | Max capital at risk per trade | `0.02` |
| `risk.max_open_positions` | Max concurrent positions | `5` |
| `risk.max_sector_exposure_pct` | Max single-sector allocation | `0.20` |
| `risk.max_portfolio_var_pct` | Max one-day 95% VaR as % of capital | `0.03` |
| `risk.risk_score_hard_stop` | Reject trade if risk score exceeds this | `80` |
| `signals.min_confidence_threshold` | Minimum confidence to emit a signal | `0.65` |
| `signals.lookback_window` | Timesteps passed to the LSTM | `60` |
| `signals.target_horizon` | Forward window used for labels | `5` |
| `execution.mode` | `paper` or live broker routing | `paper` |
| `telegram.enabled` | Enable Telegram notifications | `false` |
| `api_keys.*` | External API credentials | `REPLACE_ME` |

## Backtest results

_Placeholder sample metrics for the public repo:_

| Strategy | CAGR | Sharpe | Max Drawdown | Win Rate |
|---|---:|---:|---:|---:|
| AI Strategy | 21.4% | 1.68 | -8.7% | 57% |
| Nifty 50 | 13.2% | 0.94 | -12.4% | 49% |

## Screenshots

- `docs/screenshots/live-overview.png` _(placeholder)_
- `docs/screenshots/risk-analytics.png` _(placeholder)_
- `docs/screenshots/backtesting.png` _(placeholder)_

## Contributing

Contributions are welcome from developers, quants, data scientists, and Indian market enthusiasts.

1. Fork the repository
2. Create a feature branch
3. Add tests for your change
4. Run `python -m unittest discover -s tests`
5. Open a pull request with a clear summary

Ideas that would make this project stronger:

- Broker adapters beyond Zerodha
- Lower-latency data ingestion
- Better news and options-flow signals
- Strategy packs for intraday, swing, and positional trading

## GitHub topics

`algorithmic-trading`, `nse`, `indian-stock-market`, `ai-trading`, `risk-management`, `machine-learning`, `streamlit`, `python`, `zerodha`, `automated-trading`

## Disclaimer

**For educational purposes. Not financial advice.**

Live trading carries material financial risk. Validate every model, limit exposure, and comply with broker and exchange rules before using real capital.
