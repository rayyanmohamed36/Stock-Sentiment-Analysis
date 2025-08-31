# Script compares news headlines sentiment with stock price
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download("vader_lexicon")
# Reads news file
df = pd.read_csv("Combined_News_DJIA.csv")
# 'Date' | 'Label' | 'Top1' ... 'Top25'
# Label = 1 (DJIA up) or 0 (DJIA down)
print(df.head())
# 2. Data Cleaning
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))  # remove non-letters
    return text.lower()

df['Combined'] = df.iloc[:, 2:27].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
df['Combined'] = df['Combined'].apply(clean_text)

df['Sentiment'] = df['Combined'].apply(
    lambda x: analyzer.polarity_scores(x)['compound'] if x.strip() else 0
)


# Combine all 25 headlines into one string per day
df['Combined'] = df.iloc[:, 2:27].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
df['Combined'] = df['Combined'].apply(clean_text)
# 3. Sentiment Analysis (VADER)
analyzer = SentimentIntensityAnalyzer()
df['Sentiment'] = df['Combined'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# 4. Get Stock Price Data (Dow Jones ETF: DIA)

prices = yf.download("DIA", start=df['Date'].min(), end=df['Date'].max())
prices = prices[['Close']]
prices['Return'] = prices['Close'].pct_change()

prices = yf.download("DIA", start=df['Date'].min(), end=df['Date'].max())

if isinstance(prices.columns, pd.MultiIndex):
    prices.columns = prices.columns.get_level_values(0)

prices = prices[['Close']]
prices['Return'] = prices['Close'].pct_change()


# 5. Merge Headlines Sentiment with Stock Data

# Convert dates properly
df['Date'] = pd.to_datetime(df['Date'])
daily_sentiment = df.groupby('Date')['Sentiment'].mean().to_frame(name="Sentiment")

# Download stock data
prices = yf.download("DIA", start=df['Date'].min(), end=df['Date'].max())

# Flatten if multi-index
if isinstance(prices.columns, pd.MultiIndex):
    prices.columns = prices.columns.get_level_values(0)

prices = prices[['Close']]
prices['Return'] = prices['Close'].pct_change()

# Align on common dates only
merged = pd.merge(
    prices, daily_sentiment, 
    left_index=True, right_index=True, 
    how="inner"
)

print("Merged shape:", merged.shape)
print(merged.head())



# 6. Correlation Analysis

corr = merged['Return'].corr(merged['Sentiment'])
print("Correlation between sentiment & stock return:", corr)

# 7. Visualization

plt.figure(figsize=(10,5))
plt.scatter(merged['Sentiment'], merged['Return'], alpha=0.6)
plt.xlabel("Daily Avg Sentiment")
plt.ylabel("Daily Stock Return")
plt.title("News Headlines Sentiment vs Dow Jones Returns")
plt.grid(True)
plt.show()


# 8. Optional: Lag Analysis

merged['Lagged_Sentiment'] = merged['Sentiment'].shift(1)
lag_corr = merged['Return'].corr(merged['Lagged_Sentiment'])
print("Lagged correlation (yesterday’s sentiment vs today’s return):", lag_corr)
