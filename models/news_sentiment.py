from __future__ import annotations

import yfinance as yf

class NewsSentimentAnalyzer:
    def __init__(self):
        try:
            from transformers import pipeline
            # Using a very lightweight model to avoid OOM or slow down
            self.sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        except Exception as e:
            print(f"Failed to load transformers pipeline: {e}")
            self.sentiment_pipeline = None

    def analyze_symbol(self, symbol: str) -> dict[str, Any]:
        """Fetch latest news for symbol and calculate average sentiment."""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                # If no news, try with or without .NS
                alt_symbol = symbol.replace(".NS", "") if ".NS" in symbol else f"{symbol}.NS"
                news = yf.Ticker(alt_symbol).news
                
            if not news or not self.sentiment_pipeline:
                return {"label": "NEUTRAL", "score": 0.5, "articles_analyzed": 0}
                
            pos_score = 0.0
            neg_score = 0.0
            count = 0
            
            for item in news[:5]:  # Analyze top 5 recent news
                title = item.get("title", "")
                if not title:
                    continue
                result = self.sentiment_pipeline(title[:512])[0]
                if result["label"] == "POSITIVE":
                    pos_score += result["score"]
                else:
                    neg_score += result["score"]
                count += 1
                
            if count == 0:
                return {"label": "NEUTRAL", "score": 0.5, "articles_analyzed": 0}
                
            total_pos = pos_score / count
            total_neg = neg_score / count
            
            if total_pos > total_neg:
                return {"label": "POSITIVE", "score": total_pos, "articles_analyzed": count}
            elif total_neg > total_pos:
                return {"label": "NEGATIVE", "score": total_neg, "articles_analyzed": count}
            else:
                return {"label": "NEUTRAL", "score": 0.5, "articles_analyzed": count}
                
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            return {"label": "NEUTRAL", "score": 0.5, "articles_analyzed": 0}
