# =============================================================================
# Tyson Dailey
# Market Sector Heatmap

#type 'py -3.12 -m streamlit run test.py' in consoleon project directory
# =============================================================================
#import numpy as np
#import matplotlib.pyplot as plt
#import plotly.io as pio

import pandas as pd
import yfinance as yf
import riskfolio as rp
import plotly.express as px
import matplotlib.pyplot as plt
import statsmodels.api as sm
import streamlit as st
from darts import TimeSeries
from darts.models import AutoARIMA, ExponentialSmoothing, Prophet
from darts.metrics import mae

# =============================================================================
# streamlit configurations
# =============================================================================
st.set_page_config(layout="wide")

st.title('Sector Market Analysis Dashboard')

st.sidebar.header("Filters")

period = st.sidebar.selectbox(
    "Select time period",
    ["1D", "5D", "1M", "3M", "6M", "1Y"])

period_map = {
    "1D": "1d",
    "5D": "5d",
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "1Y": "1y"
}
period_str = period_map[period]

# =============================================================================
# Creating Sector representations
# =============================================================================

SECTORS={
    'TECH': ['AAPL','MSFT','GOOGL','NVDA','AMD','ORCL','PLTR','CSCO'],
    'FINANCIAL_SERV': ['JPM','BAC','C','V','MA','MS','GS','WFC'],
    'HEALTHCARE': ['PFE','LLY','UNH','JNJ','MRK','TMO','ABBV','CVS'],
    'CONSUMER_CYCL': ['HD','MCD','TJX','SBUX','AMZN','TSLA','LOW','ABNB'],
    'COMMUNICATION_SERV': ['TMUS','META','NFLX','VZ','T','DIS','SPOT','CMCSA'],
    'INDUSTRIALS': ['CAT','GE','BA','DE','UNP','LMT','HON','HWM'],
    'ENERGY': ['XOM','CVX','WMB','COP','SLB','VLO','MPC','EPD'],
    'CONSUMER_DEF': ['WMT','COST','PG','KO','PEP','PM','MDLZ','CL'],
    'BASIC_MATERIALS': ['LIN','SCCO','NEM','SHW','FCX','NUE','CRH','ECL'],
    'UTILITIES': ['NEE','CEG','SO','DUK','D','VST','AEP','SRE'],
    'REAL_ESTATE': ['WELL','PLD','EQIX','AMT','SPG','PSA','DLR','O']
}

DIV = 1e9 
UNIT = "USD (billions)"

data=[]

for sector, tickers in SECTORS.items():
    for ticker in tickers:
        stock= yf.Ticker(ticker)
        
        info= stock.info
        name= info.get('longName')
        market_cap= info.get('marketCap')
        
        if period == "1D":
            hist = stock.history(period="5d", interval="1d")
        else:
            hist = stock.history(period=period_str)
        if period == "1D":
            pct_change = (hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1
        else:
            pct_change = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
           
        data.append({
            'Company': name,
            'Ticker': ticker,
            'Sector': sector,
            'MarketCap': market_cap,
            'Change': pct_change
        })
    
df = pd.DataFrame(data)
df["MarketCap"] = df["MarketCap"] / DIV
df["Change"] = df["Change"].round(4)
df['Percent_Change']= df['Change']* 100

############# Toatal Market ---
total_market_cap = df["MarketCap"].sum()

# weighted average change (important — not just mean)
weighted_change = (df["MarketCap"] * df["Change"]).sum() / total_market_cap

################ create a new row for ALL sectors
df_all = pd.DataFrame([{
    'Ticker': 'ALL',
    'Sector': 'ALL',
    'MarketCap': total_market_cap,
    'Change': weighted_change
}])

############### append to main dataframe
df = pd.concat([df, df_all], ignore_index=True)

############### Separate tabs

tab1, tab2,tab3 = st.tabs(["Sector Analysis",'Sector Outlook',"Portfolio Performance"])


############### top movers ---
with tab1:
    st.subheader("Top Movers")
    
    top_gainers = df[df['Ticker'] != 'ALL'].nlargest(3, 'Change')
    top_losers = df[df['Ticker'] != 'ALL'].nsmallest(3, 'Change')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Top Gainers")
        st.dataframe(top_gainers[['Ticker', 'Percent_Change']])
    
    with col2:
        st.write("Top Losers")
        st.dataframe(top_losers[['Ticker', 'Percent_Change']])
    
    
    st.metric(
        label="Total Market Cap (All Sectors)",
        value=f"{total_market_cap:.2f}B",
        delta=f"{weighted_change:.2%}"
    )
    
    ################## Plotting Heatmap ##############
    fig = px.treemap(
        df[df['Sector'] != 'ALL'],
        path=['Sector', 'Ticker'],
        values='MarketCap',
        color='Change',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0
    )
    
    fig.update_traces(
        textinfo='label+text',
        text=df["Change"].apply(lambda x: f"{x:.2%}")
        )
    
    ##adding title to center, fizing chart height, and making the color scole whole numbers 
    fig.update_layout(
        title='Sector Market Analysis',
        title_x= .5,
        height= 700,
        coloraxis_colorbar=dict(tickformat='.0%') #Legend being whole numbers
        )
    
    #See a quick glance when hovering over the chart
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>' +
                      'Market Cap: %{value:.2f}B<br>' +
                      'Change: %{color:.2%}<extra></extra>'
    )
    
    st.plotly_chart(fig, use_container_width= True)
    
    #Puttinng the data on the app and giving it color
    st.dataframe(
        df[['Company','Ticker','Sector','MarketCap', 'Percent_Change']]
        .style.background_gradient(cmap='RdYlGn', subset=['Percent_Change'])
    )
    
# =============================================================================
# Which sector will perform the best next 24-months
# =============================================================================

with tab2:
    st.header("Sector with the most growth based off ensemble")
    
    sector_names = {
        'VGT': 'Information Technology',
        'XLF': 'Financials',
        'XLV': 'Health Care',
        'XLY': 'Consumer Discretionary',
        'XLC': 'Communication Services',
        'XLI': 'Industrials',
        'XLE': 'Energy',
        'XLP': 'Consumer Staples',
        'GDX': 'Gold Miners',
        'XLU': 'Utilities',
        'VNQ': 'Real Estate'
    }
    
    start, end= '2006-04-01', '2026-04-01'
    etfs=['VGT','XLF','XLV','XLY','XLC','XLI','XLE','XLP','GDX','XLU','VNQ']
    results=[]

    for ticker in etfs:

        temp_df= yf.download(ticker,start=start,end=end, auto_adjust= False, multi_level_index=False,
                         progress=False)
        
        temp_df= temp_df.dropna()
        
        temp_df = temp_df.resample('MS').last()
        
        st.write(f"Downloaded {len(temp_df)} months of {ticker}")
        st.write(temp_df.head())
        
        # ============================================
        # SECTION 2: Convert to Darts TimeSeries
        # ============================================
        
        # Darts requires its own TimeSeries object
        series = TimeSeries.from_dataframe(temp_df, value_cols='Adj Close')
        
        # Split: 90% train, 10% test
        train, test = series.split_before(0.8)
        
        st.write(f"**Training:** {len(train)} months")
        st.write(f"**Testing:** {len(test)} months")
        
        # ============================================
        # SECTION 3: Fit All Three Models
        # ============================================
        
        # Notice: all three models use identical .fit() syntax
        # This is the main advantage of Darts over using separate libraries
        
        model_arima = AutoARIMA(season_length=12)
        model_ets   = ExponentialSmoothing(seasonal_periods=12)
        model_prophet = Prophet(seasonality_mode='additive')
        
        #print("Fitting AutoARIMA...")
        model_arima.fit(train)
        
        #print("Fitting Exponential Smoothing...")
        model_ets.fit(train)
        
        #print("Fitting Prophet...")
        model_prophet.fit(train)
        
       # print("All models fitted.")
        
        # ============================================
        # SECTION 4: Generate Predictions
        # ============================================
        
        # Notice: all three models use identical .predict() syntax
        predictions_arima   = model_arima.predict(len(test))
        predictions_ets     = model_ets.predict(len(test))
        predictions_prophet = model_prophet.predict(len(test))
        
        # ============================================
        # SECTION 5: Compare Predictions to Actual
        # ============================================
        gr1, gr2= st.columns(2)
        with gr1:
            pr1= plt.figure(figsize=(12, 6))
            
            train.plot(label='Training Data', color='blue', linewidth=2)
            test.plot(label='Actual Prices', color='black', linewidth=3)
            
            predictions_arima.plot(label='AutoARIMA', color='red', 
                                   linestyle='--', linewidth=2)
            predictions_ets.plot(label='Exponential Smoothing', color='green', 
                                 linestyle='--', linewidth=2)
            predictions_prophet.plot(label='Prophet', color='orange', 
                                     linestyle='--', linewidth=2)
            
            plt.title('How Well Did Each Model Predict?', fontsize=14, fontweight='bold')
            plt.ylabel(f'{ticker} Price ($, VGT)')
            plt.xlabel('Date')
            plt.legend(fontsize=11)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(pr1)
        
        # ============================================
        # SECTION 6: Which Model Was Most Accurate?
        # ============================================
        
        error_arima   = mae(test, predictions_arima)
        error_ets     = mae(test, predictions_ets)
        error_prophet = mae(test, predictions_prophet)
        
        st.subheader(f"Accuracy for {ticker}. (Lower Better)")
        st.write(f"**AutoARIMA:** ${error_arima:.2f}")
        st.write(f"**Exponential Smoothing:** ${error_ets:.2f}")
        st.write(f"**Prophet:** ${error_prophet:.2f}")
       
        
        # ============================================
        # SECTION 7: Refit on Full Data & Forecast 24 Months
        # ============================================
        
        # Refit all models on the full dataset
        model_arima_full   = AutoARIMA(season_length=12)
        model_ets_full     = ExponentialSmoothing(seasonal_periods=12)
        model_prophet_full = Prophet(seasonality_mode='additive')
        
        #print("Refitting on full dataset...")
        model_arima_full.fit(series)
        model_ets_full.fit(series)
        model_prophet_full.fit(series)
        
        # Forecast next 24 months
        future_arima   = model_arima_full.predict(24)
        future_ets     = model_ets_full.predict(24)
        future_prophet = model_prophet_full.predict(24)
        
        # ============================================
        # SECTION 8: Plot Future Forecasts
        # ============================================
        
        with gr2:
            pr2= plt.figure(figsize=(12, 6))
            
            series.plot(label='Historical Prices', color='blue', linewidth=2)
            future_arima.plot(label='AutoARIMA Forecast', color='red', 
                              linestyle='--', linewidth=2.5)
            future_ets.plot(label='Exponential Smoothing Forecast', color='green', 
                            linestyle='--', linewidth=2.5)
            future_prophet.plot(label='Prophet Forecast', color='orange', 
                                linestyle='--', linewidth=2.5)
            
            plt.title('Forecasting Next 24 Months', fontsize=14, fontweight='bold')
            plt.ylabel(f'{ticker} Price ($)')
            plt.xlabel('Date')
            plt.legend(fontsize=11)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(pr2)
        
        #print("Forecasting complete!")
        
    # =============================================================================
    #     Section 9: Return Calculation
    # =============================================================================
        
        #getting current price
        current_price= series.last_value()

        # 2. Get the forecast price at the end of the period (24 months out)
        
        forecast_price = future_prophet.last_value()
        forecast_price_arima = future_arima.last_value()
        forecast_price_ets = future_ets.last_value()
        
        forecast_price= (forecast_price+forecast_price_arima+forecast_price_ets)/3
        
        # 3. Calculate Percent Change
        percent_change = ((forecast_price - current_price) / current_price) * 100
        
        results.append({'Sector': ticker, 'Return': percent_change})
        
        st.info(f"**{ticker} Summary:** Current: ${current_price:.2f} | Average Forecast: ${forecast_price:.2f} | Average Gain: {percent_change:.2f}%")
        

    # This prints the final leaderboard
    st.divider()
    st.subheader('Final Rankings')
    ranked_sectors = sorted(results, key=lambda x: x['Return'], reverse=True)
    for item in ranked_sectors:
        name = sector_names.get(item['Sector'], "Unknown")
        st.write(f"**{name} ({item['Sector']})**: {item['Return']:.2f}%")


# =============================================================================
# Performance is better than S&P so lets see how the portfolio performs.
# The questoin i am trying to answer is "does the S&P really need all those
# stocks to have the same performance? can i replicate the s&P with fewer stocks?"
# =============================================================================
with tab3:
    st.header('Portfolio Replicator Anlaysis')
        
    START = '2022-04-01' #CEG was not listed 2021
    END = '2026-04-01'
    
    ASSETS= ['AAPL','MSFT','GOOGL','NVDA','AMD','ORCL','PLTR','CSCO',
             'JPM','BAC','C','V','MA','MS','GS','WFC',
             'PFE','LLY','UNH','JNJ','MRK','TMO','ABBV','CVS',
             'HD','MCD','TJX','SBUX','AMZN','TSLA','LOW','ABNB',
             'TMUS','META','NFLX','VZ','T','DIS','SPOT','CMCSA',
             'CAT','GE','BA','DE','UNP','LMT','HON','HWM',
             'XOM','CVX','WMB','COP','SLB','VLO','MPC','EPD',
             'WMT','COST','PG','KO','PEP','PM','MDLZ','CL',
             'LIN','SCCO','NEM','SHW','FCX','NUE','CRH','ECL',
             'NEE','CEG','SO','DUK','D','VST','AEP','SRE',
             'WELL','PLD','EQIX','AMT','SPG','PSA','DLR','O']
    
    FREQ = 12
    TRADING_PERIODS = 12
    RF_ANNUAL = 0.03
    RF = (1 + RF_ANNUAL)**(1/12) - 1
    
    
    raw = yf.download(ASSETS, start=START, end=END, auto_adjust=False, progress=False)
    adj_close = raw['Adj Close'].dropna(how='all')
    monthly_prices = adj_close.resample('ME').last() 
    monthly_returns = monthly_prices.pct_change().dropna(how='all')
    monthly_returns = monthly_returns.dropna(axis=1, how='all')
    
    benchmark= 'SPY'
    spy_data= yf.download(benchmark,start=START,end=END,multi_level_index= False)
    spy_prices= spy_data['Close'].dropna(how='all')
    spy_monthly= spy_prices.resample('ME').last()
    spy_returns= spy_monthly.pct_change().dropna(how='all')
    spy_returns = spy_returns.to_frame()
    
    ################Forming the portfolio
    port= rp.Portfolio(returns= monthly_returns)
    port.assets_stats(method_mu='hist', method_cov='hist')
    mu_stock, cov_stock= port.mu, port.cov
    
    spy_port= rp.Portfolio(returns= spy_returns)
    spy_port.assets_stats(method_mu='hist', method_cov='hist')
    mu_spy, cov_spy= spy_port.mu, spy_port.cov
    
    
    ################Creating the equal weighted port.
    # w_eq= pd.Series(1/len(monthly_returns.columns),
    #                 index= monthly_returns.columns)
    # print(w_eq)
    
    # 1. Calculate the total market cap of your 88 stocks
    # We use the 'df' you created earlier that has the MarketCap column
    total_mc = df[df['Ticker'] != 'ALL']['MarketCap'].sum()

    # 2. Create the weights by dividing each stock's cap by the total
    # We map the ticker to its specific weight
    weights_dict = {}
    for index, row in df[df['Ticker'] != 'ALL'].iterrows():
        weights_dict[row['Ticker']] = row['MarketCap'] / total_mc

    # 3. Convert this into a Pandas Series that Riskfolio can read
    # We ensure the order matches your monthly_returns columns
    w_eq = pd.Series(weights_dict, index=monthly_returns.columns)

    # Display the weights so you can verify them in Streamlit
    st.write("Market Cap Weights (Top 5):")
    st.write(w_eq.sort_values(ascending=False).head(5))
    
    spy_eq= pd.Series(1/len(spy_returns.columns),
                    index= spy_returns.columns)
    print(spy_eq)
    
    
    ################ Growth of $100,000 
    eq_ret= monthly_returns.fillna(0).dot(w_eq)
    cum_eq= 100000*(1+eq_ret).cumprod()
    
    spy_ret= spy_returns.fillna(0).dot(spy_eq)
    cum_spy= 100000*(1+spy_ret).cumprod()
    
    ################ Plotting the 100,000 growth
    st.divider()
    st.subheader('Portfolio Performance: Groth of $100,000')
    
    gr3, gr4= st.columns(2)
    
    with gr3:
        fig1= plt.figure(figsize=(12, 6))
        plt.plot(cum_eq, label='Equal-Weight Portfolio', linewidth=2, color='#2E86AB')
        plt.title('Equal-Weight Stock Portfolio: Growth of $100,000', fontsize=14, fontweight='bold')
        plt.ylabel('Portfolio Value ($)')
        plt.xlabel('Date')
        plt.grid(alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        st.pyplot(fig1)
    
    with gr4:
        fig2= plt.figure(figsize=(12, 6))
        plt.plot(cum_spy, label='Equal-Weight Portfolio', linewidth=2, color='#2E86AB')
        plt.title('Equal-Weight SPY Portfolio: Growth of $100,000', fontsize=14, fontweight='bold')
        plt.ylabel('Portfolio Value ($)')
        plt.xlabel('Date')
        plt.grid(alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        st.pyplot(fig2)
    
# =============================================================================
# Beta and Alpha Comparison- SYP as a Benchmark
# =============================================================================
    
    def calculate_beta_alpha(portfolio_returns, market_returns):
        """
        Calculate Beta and Alpha using linear regression
    
        Beta : Sensitivity to market movements
        Alpha : Excess return above what Beta predicts
        """
        X= sm.add_constant(market_returns)
        model= sm.OLS(portfolio_returns, X).fit()
    
        alpha= model.params[0] # Intercept
        beta= model.params[1] # Slope
    
        return beta, alpha
    beta1, alpha1 = calculate_beta_alpha(eq_ret, spy_returns)
    
    #Displaying Alpha and Beta in streamlit
    st.subheader("Risk Metrics (vs SPY)")
    m1, m2 = st.columns(2)
    m1.metric("Beta (Market Sensitivity)", f"{beta1:.2f}")
    m2.metric("Alpha (Excess Return)", f"{alpha1:.4f}")

    






