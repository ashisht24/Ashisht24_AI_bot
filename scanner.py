import os
os.environ['CMDSTANPY_LOGGING'] = '40' 

import sys
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from prophet import Prophet
from textblob import TextBlob
from datetime import datetime, time as dtime
import time
import warnings
import logging
import urllib.request
import re
import numpy as np
import asyncio
import telegram

# --- 2026 COMPATIBILITY PATCH ---
try:
    pd.Int64Index = pd.Index
    pd.Float64Index = pd.Index
except:
    pass
# --------------------------------

warnings.filterwarnings("ignore")
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').disabled = True

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
        try:
            ticker_clean = self.ticker_symbol.split('.')[0]
            url = f"https://news.google.com/rss/search?q={ticker_clean}+stock+india&hl=en-IN&gl=IN&ceid=IN:en"
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                content = response.read().decode('utf-8')
                headlines = re.findall("<title>(.*?)</title>", content)[1:6]
            if not headlines: return 0.0, "No major news"
            def clean_h(h): return re.sub(r'\s+-\s+[^-]+$', '', h).strip()
            sentiment = sum([TextBlob(h).sentiment.polarity for h in headlines]) / len(headlines)
            return (sentiment, clean_h(headlines[0])) if abs(sentiment) > 0.05 else (sentiment, "No major news")
        except:
            return 0.0, "No major news"

    def analyze(self):
        try:
            df = self.df.copy()
            df['RSI'] = ta.rsi(df['Close'], length=14)
            bb = ta.bbands(df['Close'], length=20, std=2)
            macd = ta.macd(df['Close'])
            df = pd.concat([df, bb, macd], axis=1)
            
            latest = df.iloc[-1]
            ltp, rsi = float(latest['Close']), float(latest['RSI'])
            macd_h = float(latest.filter(like='MACDh').iloc[0])
            vol_now, vol_avg = float(latest['Volume']), df['Volume'].tail(20).mean()
            l_band, u_band = float(latest.filter(like='BBL').iloc[0]), float(latest.filter(like='BBU').iloc[0])

            df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
            df_p['ds'] = df_p['ds'].dt.tz_localize(None)
            m = Prophet(daily_seasonality=False, yearly_seasonality=False, weekly_seasonality=True)
            m.fit(df_p)
            forecast = m.predict(m.make_future_dataframe(periods=7))
            ai_target = float(forecast['yhat'].iloc[-1])

            sentiment, headline = self.get_news_data()
            news_label = "POSITIVE" if sentiment > 0.05 else ("NEGATIVE" if sentiment < -0.05 else "NEUTRAL")
            pe = self.info.get('forwardPE', 100)
            debt_eq = self.info.get('debtToEquity', 0) / 100
            fundamental_ok = pe < 85 and debt_eq < 2.5

            call, target, sl = "HOLD", "—", "—"
            tech_ok = (rsi < 60 and macd_h > 0 and vol_now > vol_avg * 0.8)

            if ai_target > (ltp * 1.02) and (ltp <= (l_band * 1.02) or tech_ok) and fundamental_ok:
                call, target, sl = "★ BUY ★", round(max(ai_target, u_band), 2), round(ltp * 0.96, 2)
            elif ai_target < (ltp * 0.98) and (rsi > 68 or news_label == "NEGATIVE"):
                call, target, sl = "EXIT", round(min(ai_target, l_band), 2), "—"
            else:
                if ai_target > ltp:
                    call, target, sl = "HOLD", round(max(ai_target, u_band), 2), round(ltp * 0.975, 2)
                else: return None

            return {
                "Ticker": self.ticker_symbol.replace(".NS",""), "CMP": round(ltp, 2), 
                "Call": call, "Target": target, "SL": sl, "AI_View": "Bullish" if ai_target > ltp else "Bearish",
                "News": news_label, "Headline": headline
            }
        except: return None

async def send_telegram_msg(message):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        bot = telegram.Bot(token=token)
        async with bot:
            await bot.send_message(text=message, chat_id=chat_id, parse_mode='Markdown')

def run_scanner():
    results = []
    print(f"🚀 SCANNING NIFTY 50...")
    for ticker in NIFTY_50_TICKERS:
        try:
            oracle = GrandOracle(ticker)
            report = oracle.analyze()
            if report: results.append(report)
        except: continue

    if results:
        final_df = pd.DataFrame(results)
        rank = {"★ BUY ★": 0, "HOLD": 1, "EXIT": 2}
        final_df['Sort'] = final_df['Call'].map(rank)
        sorted_results = final_df.sort_values('Sort').to_dict('records')

        cards = []
        for row in sorted_results:
            # Build the card matching your image exactly
            card = (
                f"{row['Call']} | *{row['Ticker']}*\n"
                f"💰 **CMP:** ₹{row['CMP']}\n"
                f"🎯 **Target:** ₹{row['Target']}\n"
            )
            
            # Use 'Take_Profit' label for EXIT calls as per your image
            if row['Call'] == "EXIT":
                card += f"💵 **Take_Profit:** ₹{row['CMP']} (Book Now!)\n"
            else:
                card += f"🛡️ **SL:** ₹{row['SL']}\n"

            card += (
                f"💡 **AI View:** {row['AI_View']}\n"
                f"📢 **News:** {row['News']}\n"
                f"📰 **Headline:** {row['Headline']}\n"
            )
            cards.append(card)

        # Final footer with timestamp
        footer = f"\nScan completed at {datetime.now().strftime('%H:%M:%S')} PM IST.\nNext scan in 30 minutes..."
        full_message = "\n\n".join(cards) + footer
        
        asyncio.run(send_telegram_msg(full_message))
        print("Report sent to Telegram.")
    else:
        print("No significant trends detected.")

if __name__ == "__main__":
    run_scanner()
