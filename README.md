# News Headlines Sentiment vs. Stock Market Returns

This project investigates the relationship between **financial news sentiment** and **stock market performance**. Using historical news headlines, we apply **Natural Language Processing (NLP)** techniques to compute daily sentiment scores and analyze their alignment with **Dow Jones Industrial Average (DIA) daily returns**.

## Project Overview

- **Objective**: Examine whether the tone of financial news can provide insight into, or help predict, stock market movements.  
- **Dataset**: Historical collection of financial news headlines, aggregated per day.  
- **Sentiment Analysis**: Headlines are processed using the **VADER sentiment analyzer** to derive daily average sentiment scores.  
- **Stock Data**: Historical prices for the Dow Jones ETF (**DIA**) are retrieved using `yfinance`.  
- **Analysis**: Sentiment scores and stock returns are merged to explore correlations and potential predictive relationships.

## Tools and Libraries

- Python  
- [NLTK](https://www.nltk.org/) – VADER sentiment analysis  
- [yfinance](https://pypi.org/project/yfinance/) – Financial data retrieval  
- [pandas](https://pandas.pydata.org/) – Data manipulation and aggregation  
- [matplotlib](https://matplotlib.org/) – Visualization

## Key Features

- Data cleaning and preprocessing of news headlines  
- Computation of daily sentiment scores (positive, negative, neutral)  
- Calculation of daily stock returns  
- Merging of sentiment and returns data for correlation analysis  
- Visualization of sentiment vs. returns through scatter plots and distribution charts  
- Support for lag analysis (previous day’s sentiment vs. current day’s returns)

## Example Result

Scatter plot showing **Daily Average Sentiment vs. Dow Jones Daily Returns**:

![Sentiment vs Returns](example_plot.png)

## Insights

- Sentiment values predominantly cluster around neutral (0)  
- Extreme negative values often result from empty or poorly processed headlines; these cases are handled in preprocessing  
- Initial correlations are generally weak, but lagged sentiment analysis can provide insights into potential predictive patterns

## Future Work

- Extend analysis to **other indices or individual stocks** (e.g., S&P 500, NASDAQ)  
- Apply **advanced NLP models** (such as BERT or FinBERT) for finance-specific sentiment detection  
- Explore longer time frames (weekly or monthly aggregation)  
- Develop a **trading strategy** informed by sentiment signals

---

**Disclaimer:** This project is intended for research and educational purposes only and should not be considered financial advice.

