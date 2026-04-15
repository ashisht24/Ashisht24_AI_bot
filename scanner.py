import os
os.environ['CMDSTANPY_LOGGING'] = '40' # Suppress heavy logging

import sys
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from prophet import Prophet
from textblob import TextBlob
from datetime import datetime, time as dtime
import time
import os
import warnings
import logging
import urllib.request
import re

import pandas as pd
import numpy as np

# --- 2026 COMPATIBILITY PATCH ---
# This fixes the 'AttributeError' in technical analysis libraries
try:
    pd.Int64Index = pd.Index
    pd.Float64Index = pd.Index
except:
    pass
# --------------------------------

# 1. Setup & Silencing Logs
warnings.filterwarnings("ignore")
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').disabled = True

# Adjusting Pandas display
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.expand_frame_repr', False)

NIFTY_50_TICKERS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "BPCL.NS", "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS",
    "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS",
    "INDUSINDBK.NS", "INFY.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS",
    "LT.NS", "LTIM.NS", "M&M.NS", "MARUTI.NS", "NESTLEIND.NS",
    "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS",
    "SBIN.NS", "SUNPHARMA.NS", "TATACONSUM.NS", "TATASTEEL.NS",
    "TCS.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "SUZLON.NS"
]

class GrandOracle:
    def __init__(self, ticker):
        self.ticker_symbol = ticker
        self.stock = yf.Ticker(ticker)
        self.df = self.stock.history(period="1y")
        if self.df.empty: raise ValueError("No data")
        self.info = self.stock.info

    def get_news_data(self):
        """Fetches sentiment and full cleaned headlines"""
        try:
            ticker_clean = self.ticker_symbol.split('.')[0]
            url = f"https://news.google.com/rss/search?q={ticker_clean}+stock+india&hl=en-IN&gl=IN&ceid=IN:en"
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                content = response.read().decode('utf-8')
                headlines = re.findall("<title>(.*?)</title>", content)[1:6]
            
            if not headlines: return 0.0, "No major news"
            
            def clean_h(h):
                return re.sub(r'\s+-\s+[^-]+$', '', h).strip()

            sentiment = sum([TextBlob(h).sentiment.polarity for h in headlines]) / len(headlines)
            
            if abs(sentiment) > 0.05:
                return sentiment, clean_h(headlines[0])
            else:
                return sentiment, "No major news"
        except:
            return 0.0, "No major news"

    def analyze(self):
        df = self.df.copy()
        
        # Pillar 1: Technicals
        df['RSI'] = ta.rsi(df['Close'], length=14)
        bb = ta.bbands(df['Close'], length=20, std=2)
        macd = ta.macd(df['Close'])
        df = pd.concat([df, bb, macd], axis=1)
        
        latest = df.iloc[-1]
        ltp = float(latest['Close'])
        rsi = float(latest['RSI'])
        macd_h = float(latest.filter(like='MACDh').iloc[0])
        vol_now = float(latest['Volume'])
        vol_avg = df['Volume'].tail(20).mean()
        l_band = float(latest.filter(like='BBL').iloc[0])
        u_band = float(latest.filter(like='BBU').iloc[0])

        # Pillar 2: AI Forecast
        df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
        df_p['ds'] = df_p['ds'].dt.tz_localize(None)
        m = Prophet(daily_seasonality=False, yearly_seasonality=False, weekly_seasonality=True)
        m.fit(df_p)
        forecast = m.predict(m.make_future_dataframe(periods=7))
        ai_target = float(forecast['yhat'].iloc[-1])

        # Pillar 3: News
        sentiment, headline = self.get_news_data()
        news_label = "POSITIVE" if sentiment > 0.05 else ("NEGATIVE" if sentiment < -0.05 else "NEUTRAL")
        
        # Fundamentals
        pe = self.info.get('forwardPE', 100)
        debt_eq = self.info.get('debtToEquity', 0) / 100
        fundamental_ok = pe < 85 and debt_eq < 2.5

        call, target, sl, tp = "HOLD", "—", "—", "—"

        # Technical Weighted Check
        tech_ok = (rsi < 60 and macd_h > 0 and vol_now > vol_avg * 0.8)

        if ai_target > (ltp * 1.02) and (ltp <= (l_band * 1.02) or tech_ok) and fundamental_ok:
            call = "★ BUY ★"
            target = round(max(ai_target, u_band), 2)
            sl = round(ltp * 0.96, 2)
        elif ai_target < (ltp * 0.98) and (rsi > 68 or news_label == "NEGATIVE"):
            call = "EXIT"
            target = round(min(ai_target, l_band), 2)
            tp = round(ltp, 2)
        else:
            if ai_target > ltp:
                call = "HOLD"
                target = round(max(ai_target, u_band), 2)
                sl = round(ltp * 0.975, 2)
            else:
                return None

        return {
            "Ticker": self.ticker_symbol.replace(".NS",""), 
            "CMP": round(ltp, 2), 
            "Call": call, 
            "Target": target, 
            "SL": sl,
            "AI_View": "BULLISH" if ai_target > ltp else "BEARISH", # RESTORED
            "News": news_label,
            "Headline": headline,
            "P/E": round(pe, 1)
        }

def is_market_open():
    now = datetime.now().time()
    return (datetime.now().weekday() < 5) and (dtime(9, 15) <= now <= dtime(15, 30))

def run_scanner():
    # ADD THIS LINE RIGHT HERE at the start of the function:
    results = [] 
    
    # ... your existing code that loops through tickers ...
    # for ticker in tickers:
    #     analysis = oracle.analyze(ticker)
    #     if analysis:
    #         results.append(analysis)

    # Now when the code reaches line 160, 'results' will exist (even if it's empty)
    if results:
        message = "🚀 **Nifty 50 Analysis Report** 🚀\n\n"
        # ... rest of your code ...

    if results:
        final_df = pd.DataFrame(results)
        
        # Create a single alert string to avoid multiple async calls
        alerts = []
        for _, row in final_df.iterrows():
            if "★" in row['Call'] or row['Call'] == "EXIT":
                msg = f"📢 *{row['Ticker']}*: {row['Call']} @ {row['CMP']}"
                alerts.append(msg)
        
        if alerts:
            full_message = "\n".join(alerts)
            # Run the message sender ONCE
            asyncio.run(send_telegram_msg(full_message))
        
        print(final_df.to_string(index=False))

if __name__ == "__main__":
    # Ensure the script actually runs
    run_scanner()
    
    import os
TELEGRAM_TOKEN = os.getenv("8680250643:AAHw2trtVMg9X5beNlcEaWjgwnFtIxw5F-Q")
CHAT_ID = os.getenv("8618622359")
