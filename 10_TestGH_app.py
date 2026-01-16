# dashboard_forex_pro_no_textblob.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Forex Pro Dashboard Premium",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with dark theme only
st.markdown("""
<style>
    :root {
        --bg-primary: #0F172A;
        --bg-secondary: #1E293B;
        --bg-card: #1E293B;
        --text-primary: #F1F5F9;
        --text-secondary: #94A3B8;
        --border-color: #334155;
        --accent-color: #60A5FA;
        --positive-color: #34D399;
        --negative-color: #F87171;
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2);
    }
    
    .main-header {
        font-size: 2.5rem;
        color: var(--accent-color);
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 600;
    }
    
    .dashboard-card {
        background-color: var(--bg-card);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: var(--shadow);
        border: 1px solid var(--border-color);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        color: var(--text-primary);
    }
    
    .dashboard-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    .card-title {
        font-size: 1.1rem;
        color: var(--text-primary);
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0.2rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .positive-change {
        color: var(--positive-color);
        font-weight: 600;
    }
    
    .negative-change {
        color: var(--negative-color);
        font-weight: 600;
    }
    
    .pair-chip {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        background: var(--bg-secondary);
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 0.2rem;
        border: 1px solid var(--border-color);
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .pair-chip:hover {
        background: var(--accent-color);
        color: white;
    }
    
    .pair-chip.active {
        background: var(--accent-color);
        color: white;
    }
    
    /* Tab styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: var(--bg-secondary);
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 4px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        color: var(--text-secondary);
        border: 1px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(96, 165, 250, 0.1);
        color: var(--accent-color);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--accent-color) !important;
        color: white !important;
        border-color: var(--accent-color);
    }
    
    .stTabs [aria-selected="true"]:hover {
        background-color: var(--accent-color) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Main Forex pairs list
FOREX_PAIRS = {
    'EURUSD=X': 'EUR/USD',
    'GBPUSD=X': 'GBP/USD', 
    'USDJPY=X': 'USD/JPY',
    'USDCHF=X': 'USD/CHF',
    'AUDUSD=X': 'AUD/USD',
    'USDCAD=X': 'USD/CAD',
    'NZDUSD=X': 'NZD/USD',
    'EURGBP=X': 'EUR/GBP',
    'EURJPY=X': 'EUR/JPY',
    'GBPJPY=X': 'GBP/JPY',
    'AUDJPY=X': 'AUD/JPY',
    'EURCHF=X': 'EUR/CHF',
    'GBPCHF=X': 'GBP/CHF',
    'CADJPY=X': 'CAD/JPY',
    'CHFJPY=X': 'CHF/JPY',
    'EURCAD=X': 'EUR/CAD',
    'EURAUD=X': 'EUR/AUD',
    'GBPAUD=X': 'GBP/AUD',
    'AUDCAD=X': 'AUD/CAD',
    'AUDCHF=X': 'AUD/CHF'
}

# Session state initialization
def init_session_state():
    defaults = {
        'selected_pair': 'EURUSD=X',
        'timeframe': '1h',
        'periods': 200,
        'show_sma': True,
        'show_ema': True,
        'show_rsi': True,
        'show_macd': True,
        'show_bollinger': True,
        'sma_periods': [20, 50, 200],
        'ema_periods': [9, 21, 50],
        'multi_pairs': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'EURGBP=X'],
        'correlation_period': '12m',
        'selected_corr_pairs': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'EURGBP=X', 'USDCHF=X']
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Function to get forex data
@st.cache_data(ttl=60)
def get_forex_data(pair, timeframe='1h', periods=200):
    try:
        period_map = {
            '1m': '7d', '5m': '60d', '15m': '60d', '30m': '60d',
            '1h': '730d', '4h': '730d', '1d': '5y'
        }
        
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '4h': '1h', '1d': '1d'
        }
        
        period = period_map.get(timeframe, '60d')
        interval = interval_map.get(timeframe, '1h')
        
        ticker = yf.Ticker(pair)
        data = ticker.history(period=period, interval=interval)
        
        if len(data) > periods:
            data = data[-periods:]
        
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        data.columns = ['open', 'high', 'low', 'close', 'volume']
        data.index.name = 'date'
        
        return data.dropna()
    
    except Exception as e:
        return pd.DataFrame()

# Function to get 1-year historical data
@st.cache_data(ttl=300)
def get_1year_data(pair):
    """Get 1-year historical daily data"""
    try:
        ticker = yf.Ticker(pair)
        data = ticker.history(period='1y', interval='1d')
        
        if not data.empty:
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            data.index.name = 'date'
            
            return data.dropna()
        else:
            return pd.DataFrame()
            
    except Exception as e:
        return pd.DataFrame()

# Function to get historical data for 1-year performance
@st.cache_data(ttl=300)
def get_1year_performance_data(pair):
    """Get 1-year historical data for performance comparison"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        ticker = yf.Ticker(pair)
        data = ticker.history(start=start_str, end=end_str, interval='1d')
        
        if len(data) == 0:
            # Try with period if date range fails
            data = ticker.history(period='1y', interval='1d')
        
        if not data.empty:
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            data.index.name = 'date'
            
            # Normalize to base 100 for comparison
            if len(data) > 0:
                first_price = data['close'].iloc[0]
                data['normalized'] = (data['close'] / first_price) * 100
            
            return data.dropna()
        else:
            return pd.DataFrame()
            
    except Exception as e:
        return pd.DataFrame()

# Fonction pour créer le graphique des courbes des paires choisies sur 1 an
def create_1year_pair_curves_chart(selected_pairs):
    """Create 1-year chart with price curves for selected pairs (normalized)"""
    
    try:
        if len(selected_pairs) < 2:
            st.warning("Please select at least 2 pairs for comparison.")
            return None
        
        # Get 1-year data for all pairs
        data_dict = {}
        available_pairs = []
        
        for pair in selected_pairs:  # Pas de limite - on peut avoir plus de paires
            data = get_1year_data(pair)
            if not data.empty and len(data) > 50:
                data_dict[pair] = data
                available_pairs.append(pair)
        
        if len(available_pairs) < 2:
            st.warning("Insufficient data for comparison.")
            return None
        
        # Find common dates
        common_dates = None
        for pair, data in data_dict.items():
            if common_dates is None:
                common_dates = data.index
            else:
                common_dates = common_dates.intersection(data.index)
        
        if len(common_dates) < 50:
            st.warning("Not enough common dates between pairs.")
            return None
        
        # Filter data to keep only common dates
        filtered_data = {}
        for pair, data in data_dict.items():
            filtered_data[pair] = data.loc[common_dates]
        
        # Create the chart
        fig = go.Figure()
        
        # Colors for different pairs - plus de couleurs pour plus de paires
        colors = ['#60A5FA', '#34D399', '#F87171', '#FBBF24', '#8B5CF6', '#EC4899', 
                  '#10B981', '#F59E0B', '#6366F1', '#EF4444', '#14B8A6', '#F97316']
        
        # Add trace for each pair (normalized to starting price = 100)
        for idx, pair in enumerate(available_pairs):
            pair_name = FOREX_PAIRS.get(pair, pair)
            data = filtered_data[pair]
            
            # Normalize prices to start at 100
            first_price = data['close'].iloc[0]
            normalized_prices = (data['close'] / first_price) * 100
            
            # Calculate total return
            total_return = ((data['close'].iloc[-1] - first_price) / first_price) * 100
            
            # Calculate additional metrics
            max_price = data['close'].max()
            min_price = data['close'].min()
            max_return = ((max_price - first_price) / first_price) * 100
            min_return = ((min_price - first_price) / first_price) * 100
            
            # Create hover text with more information
            hover_text = (f'{pair_name}<br>'
                         f'Date: %{{x|%Y-%m-%d}}<br>'
                         f'Normalized Price: %{{y:.2f}}<br>'
                         f'Total Return: {total_return:+.2f}%<br>'
                         f'Max Return: {max_return:+.2f}%<br>'
                         f'Min Return: {min_return:+.2f}%<br>'
                         f'Current Price: {data["close"].iloc[-1]:.5f}')
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=normalized_prices,
                    name=f'{pair_name} ({total_return:+.1f}%)',
                    line=dict(color=colors[idx % len(colors)], width=2.5),
                    mode='lines',
                    hovertemplate=hover_text + '<extra></extra>'
                )
            )
        
        # Update layout
        fig.update_layout(
            title='1-Year Pair Price Comparison (Normalized to 100)',
            height=550,
            template='plotly_dark',
            hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#F1F5F9',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,  # Mettre la légende en bas
                xanchor="center",
                x=0.5,
                bgcolor='rgba(30, 41, 59, 0.8)',
                bordercolor='#334155',
                borderwidth=1
            ),
            margin=dict(b=100),  # Marge en bas pour la légende
            xaxis=dict(
                title='Date',
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                tickformat='%b %Y'
            ),
            yaxis=dict(
                title='Normalized Price (Base 100)',
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)'
            )
        )
        
        # Add summary statistics annotation
        if len(available_pairs) > 0:
            stats_text = "<b>Summary (1-Year):</b><br>"
            for idx, pair in enumerate(available_pairs[:5]):  # Limiter à 5 pour ne pas surcharger
                pair_name = FOREX_PAIRS.get(pair, pair)
                data = filtered_data[pair]
                total_return = ((data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]) * 100
                stats_text += f"{pair_name}: {total_return:+.1f}%<br>"
            
            if len(available_pairs) > 5:
                stats_text += f"... and {len(available_pairs) - 5} more pairs"
            
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                text=stats_text,
                showarrow=False,
                align="left",
                font=dict(color="#94A3B8", size=11),
                bgcolor="rgba(30, 41, 59, 0.8)",
                bordercolor="#334155",
                borderwidth=1,
                borderpad=6
            )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating 1-year pair curves chart: {str(e)}")
        return None

# Function to create correlation matrix
def create_correlation_matrix(pairs_list, timeframe='1h', periods=100):
    """Create correlation matrix between multiple pairs"""
    close_prices = pd.DataFrame()
    
    for pair in pairs_list:
        data = get_forex_data(pair, timeframe, periods)
        if not data.empty:
            close_prices[FOREX_PAIRS.get(pair, pair)] = data['close']
    
    if not close_prices.empty:
        # Calculate returns for correlation
        returns = close_prices.pct_change().dropna()
        
        if len(returns) > 5:  # Minimum data
            correlation_matrix = returns.corr()
            return correlation_matrix, close_prices
        else:
            return pd.DataFrame(), close_prices
    
    return pd.DataFrame(), pd.DataFrame()

# Function to create multi-pair view
def create_multi_pair_view(selected_pairs, timeframe='1h'):
    """Create compact view for multiple pairs"""
    st.markdown("### 📊 Multi-Pair View")
    
    cols = st.columns(len(selected_pairs))
    
    for idx, pair in enumerate(selected_pairs):
        with cols[idx]:
            data = get_forex_data(pair, timeframe, periods=50)
            
            if not data.empty:
                current_price = data['close'].iloc[-1]
                prev_price = data['close'].iloc[-2] if len(data) > 1 else current_price
                price_change = ((current_price - prev_price) / prev_price) * 100
                
                # Compact card
                st.markdown(f"""
                <div class="dashboard-card" style="padding: 1rem;">
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">
                        {FOREX_PAIRS.get(pair, pair)}
                    </div>
                    <div style="font-size: 1.4rem; font-weight: 700; margin-bottom: 0.3rem;">
                        {current_price:.5f}
                    </div>
                    <div style="color: {'#34D399' if price_change > 0 else '#F87171'}; font-weight: 600;">
                        {'+' if price_change > 0 else ''}{price_change:.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Mini sparkline chart
                fig = go.Figure(data=go.Scatter(
                    x=data.index,
                    y=data['close'],
                    mode='lines',
                    line=dict(width=2, color='#60A5FA'),
                    fill='tozeroy',
                    fillcolor='rgba(96, 165, 250, 0.1)'
                ))
                
                fig.update_layout(
                    height=80,
                    margin=dict(l=0, r=0, t=0, b=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, zeroline=False, visible=False),
                    yaxis=dict(showgrid=False, zeroline=False, visible=False),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# Function to create 1-year performance comparison chart
def create_1year_performance_chart(selected_pairs):
    """Create chart showing 1-year performance comparison for selected pairs"""
    
    try:
        if len(selected_pairs) < 2:
            st.info("Select at least 2 pairs for performance comparison.")
            return None
        
        # Get 1-year data for all selected pairs
        performance_data = {}
        available_pairs = []
        
        for pair in selected_pairs[:6]:  # Limit to 6 pairs for readability
            data = get_1year_performance_data(pair)
            if not data.empty and 'normalized' in data.columns:
                performance_data[pair] = data
                available_pairs.append(pair)
        
        if len(available_pairs) < 2:
            st.warning("Insufficient data for performance comparison.")
            return None
        
        # Create the chart
        fig = go.Figure()
        
        # Colors for different pairs
        colors = ['#60A5FA', '#34D399', '#F87171', '#FBBF24', '#8B5CF6', '#EC4899']
        
        # Add trace for each pair
        for idx, pair in enumerate(available_pairs):
            pair_name = FOREX_PAIRS.get(pair, pair)
            data = performance_data[pair]
            
            # Calculate total return
            total_return = ((data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['normalized'],
                    name=f'{pair_name} ({total_return:+.1f}%)',
                    line=dict(color=colors[idx % len(colors)], width=3),
                    mode='lines',
                    hovertemplate=f'{pair_name}<br>Date: %{{x|%Y-%m-%d}}<br>Performance: %{{y:.1f}}%<extra></extra>'
                )
            )
        
        # Update layout
        fig.update_layout(
            title='1-Year Performance Comparison (Normalized to 100)',
            height=500,
            template='plotly_dark',
            hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#F1F5F9',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                title='Date',
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                tickformat='%b %Y'
            ),
            yaxis=dict(
                title='Performance (%)',
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)'
            )
        )
        
        # Add performance metrics table
        st.markdown("#### 📈 1-Year Performance Metrics")
        
        # Create metrics dataframe
        metrics_data = []
        for pair in available_pairs:
            data = performance_data[pair]
            pair_name = FOREX_PAIRS.get(pair, pair)
            
            # Calculate metrics
            total_return = ((data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]) * 100
            max_price = data['close'].max()
            min_price = data['close'].min()
            current_price = data['close'].iloc[-1]
            volatility = data['close'].pct_change().std() * np.sqrt(252) * 100
            
            metrics_data.append({
                'Pair': pair_name,
                'Current Price': f'{current_price:.5f}',
                'Total Return': f'{total_return:+.2f}%',
                'High': f'{max_price:.5f}',
                'Low': f'{min_price:.5f}',
                'Volatility': f'{volatility:.1f}%'
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Display metrics in columns
        cols = st.columns(len(available_pairs))
        for idx, pair in enumerate(available_pairs):
            with cols[idx]:
                data = performance_data[pair]
                pair_name = FOREX_PAIRS.get(pair, pair)
                total_return = ((data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]) * 100
                
                st.markdown(f"""
                <div class="dashboard-card" style="padding: 1rem; text-align: center;">
                    <div style="font-weight: 600; margin-bottom: 0.5rem; font-size: 0.9rem;">
                        {pair_name}
                    </div>
                    <div style="font-size: 1.8rem; font-weight: 700; margin-bottom: 0.3rem; color: {'#34D399' if total_return > 0 else '#F87171'};">
                        {total_return:+.1f}%
                    </div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary);">
                        1-Year Return
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating performance chart: {str(e)}")
        return None

# Sidebar
with st.sidebar:
    st.markdown("### 💱 Forex Pro Dashboard")
    
    # Multi-pair selection
    st.markdown("#### 📊 Pairs to Monitor")
    
    all_pairs = list(FOREX_PAIRS.keys())
    selected_multi_pairs = st.multiselect(
        "Select up to 8 pairs:",
        all_pairs,
        default=st.session_state.multi_pairs[:4],
        format_func=lambda x: FOREX_PAIRS[x],
        max_selections=8
    )
    
    if selected_multi_pairs != st.session_state.multi_pairs:
        st.session_state.multi_pairs = selected_multi_pairs
    
    # Main pair selection
    st.markdown("---")
    st.markdown("#### 🎯 Main Pair")
    
    selected_pair = st.selectbox(
        "Main pair:",
        all_pairs,
        index=all_pairs.index(st.session_state.selected_pair),
        format_func=lambda x: FOREX_PAIRS[x]
    )
    
    if selected_pair != st.session_state.selected_pair:
        st.session_state.selected_pair = selected_pair
    
    # Timeframe
    st.markdown("---")
    st.markdown("#### ⏰ Timeframe")
    timeframe = st.selectbox(
        "Interval:",
        ['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
        index=4
    )
    
    if timeframe != st.session_state.timeframe:
        st.session_state.timeframe = timeframe
    
    # Indicators
    st.markdown("---")
    st.markdown("#### 📈 Indicators")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.show_sma = st.checkbox("SMA", st.session_state.show_sma)
        st.session_state.show_ema = st.checkbox("EMA", st.session_state.show_ema)
        st.session_state.show_rsi = st.checkbox("RSI", st.session_state.show_rsi)
    
    with col2:
        st.session_state.show_macd = st.checkbox("MACD", st.session_state.show_macd)
        st.session_state.show_bollinger = st.checkbox("Bollinger", st.session_state.show_bollinger)

# Main interface
st.markdown('<h1 class="main-header">💱 Forex Pro Dashboard Premium</h1>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Multi-Pair View", "📈 Technical Analysis", "🔗 Correlation Analysis"])

with tab1:
    # Section 1: Multi-Pair View
    if st.session_state.multi_pairs:
        create_multi_pair_view(st.session_state.multi_pairs, st.session_state.timeframe)
        
        # Add 1-Year Performance Chart
        st.markdown("---")
        st.markdown("### 📈 1-Year Performance Comparison")
        
        perf_chart = create_1year_performance_chart(st.session_state.multi_pairs)
        if perf_chart:
            st.plotly_chart(perf_chart, use_container_width=True)
        else:
            st.info("Loading 1-year performance data...")
    else:
        st.info("Please select pairs in the sidebar to display the multi-pair view.")

with tab2:
    # Section 2: Technical analysis of main pair
    data = get_forex_data(st.session_state.selected_pair, st.session_state.timeframe, st.session_state.periods)
    
    if not data.empty and len(data) > 10:
        # Calculate indicators
        df = data.copy()
        
        if st.session_state.show_sma:
            for period in st.session_state.sma_periods:
                df[f'SMA_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
        
        if st.session_state.show_ema:
            for period in st.session_state.ema_periods:
                df[f'EMA_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
        
        if st.session_state.show_rsi:
            df['RSI'] = ta.momentum.rsi(df['close'], window=14)
        
        if st.session_state.show_macd:
            macd = ta.trend.MACD(df['close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_diff'] = df['MACD'] - df['MACD_signal']
        
        if st.session_state.show_bollinger:
            bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['BB_upper'] = bollinger.bollinger_hband()
            df['BB_middle'] = bollinger.bollinger_mavg()
            df['BB_lower'] = bollinger.bollinger_lband()
        
        # Main metrics
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
        price_change = ((current_price - prev_price) / prev_price) * 100
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="dashboard-card">
                <div class="card-title">💰 Current Price</div>
                <div class="metric-value">
                    {current_price:.5f}
                </div>
                <div class="{'positive-change' if price_change > 0 else 'negative-change'}">
                    {'+' if price_change > 0 else ''}{price_change:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            daily_range = df['high'].iloc[-1] - df['low'].iloc[-1]
            avg_range = (df['high'] - df['low']).mean()
            range_change = ((daily_range - avg_range) / avg_range) * 100
            st.markdown(f"""
            <div class="dashboard-card">
                <div class="card-title">📏 Daily Range</div>
                <div class="metric-value">
                    {daily_range:.5f}
                </div>
                <div class="{'positive-change' if range_change > 0 else 'negative-change'}">
                    {'+' if range_change > 0 else ''}{range_change:.2f}% vs avg
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].mean()
            volume_change = ((volume - avg_volume) / avg_volume) * 100
            st.markdown(f"""
            <div class="dashboard-card">
                <div class="card-title">📈 Volume</div>
                <div class="metric-value">
                    {volume:,.0f}
                </div>
                <div class="{'positive-change' if volume_change > 0 else 'negative-change'}">
                    {'+' if volume_change > 0 else ''}{volume_change:.2f}% vs avg
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            volatility = df['close'].pct_change().std() * np.sqrt(252) * 100
            st.markdown(f"""
            <div class="dashboard-card">
                <div class="card-title">🌪️ Annual Volatility</div>
                <div class="metric-value">
                    {volatility:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            if 'RSI' in df.columns:
                rsi = df['RSI'].iloc[-1]
                rsi_signal = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"
                color = "#F87171" if rsi > 70 else "#34D399" if rsi < 30 else "#94A3B8"
                st.markdown(f"""
                <div class="dashboard-card">
                    <div class="card-title">📊 RSI (14)</div>
                    <div class="metric-value" style="color: {color};">
                        {rsi:.1f}
                    </div>
                    <div style="font-weight: 600; color: {color};">
                        {rsi_signal}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Main chart
        st.markdown("### 📈 Main Chart")
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{FOREX_PAIRS.get(st.session_state.selected_pair, st.session_state.selected_pair)} - Technical Analysis', 'Volume')
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='#34D399',
                decreasing_line_color='#F87171'
            ),
            row=1, col=1
        )
        
        # Indicators
        if st.session_state.show_sma:
            for period in st.session_state.sma_periods[:2]:
                if f'SMA_{period}' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[f'SMA_{period}'],
                            name=f'SMA {period}',
                            line=dict(width=1.5),
                            opacity=0.7
                        ),
                        row=1, col=1
                    )
        
        if st.session_state.show_bollinger and 'BB_upper' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_upper'],
                    name='BB Upper',
                    line=dict(color='rgba(96, 165, 250, 0.3)', width=1),
                    showlegend=False
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_lower'],
                    name='BB Lower',
                    line=dict(color='rgba(96, 165, 250, 0.3)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(96, 165, 250, 0.1)',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Volume
        colors = ['#34D399' if df['close'].iloc[i] >= df['open'].iloc[i] else '#F87171' 
                  for i in range(len(df))]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.6
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#F1F5F9'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Nouvelles sections de Technical Analysis (remplacement de Technical Sentiment Analysis)
        st.markdown("---")
        st.markdown("### 📊 Technical Overview")
        
        col_tech1, col_tech2 = st.columns(2)
        
        with col_tech1:
            # Price Analysis
            st.markdown(f"""
            <div class="dashboard-card">
                <div class="card-title">📈 Price Analysis</div>
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span>Price vs SMA 20:</span>
                        <span style="font-weight: 600; color: {'#34D399' if current_price > df['SMA_20'].iloc[-1] else '#F87171'};">
                            {'+' if current_price > df['SMA_20'].iloc[-1] else ''}{(current_price - df['SMA_20'].iloc[-1]):.5f}
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span>Price vs EMA 9:</span>
                        <span style="font-weight: 600; color: {'#34D399' if current_price > df['EMA_9'].iloc[-1] else '#F87171'};">
                            {'+' if current_price > df['EMA_9'].iloc[-1] else ''}{(current_price - df['EMA_9'].iloc[-1]):.5f}
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span>ATR (14):</span>
                        <span style="font-weight: 600;">
                            {ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14).iloc[-1]:.5f}
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Today's Range:</span>
                        <span style="font-weight: 600;">
                            {daily_range:.5f}
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Trend Analysis
            if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
                sma_20 = df['SMA_20'].iloc[-1]
                sma_50 = df['SMA_50'].iloc[-1]
                trend_status = ""
                
                if current_price > sma_20 > sma_50:
                    trend_status = "STRONG UPTREND 🟢"
                    trend_color = "#34D399"
                elif current_price > sma_20 and sma_20 > sma_50:
                    trend_status = "UPTREND 🟢"
                    trend_color = "#34D399"
                elif current_price < sma_20 < sma_50:
                    trend_status = "STRONG DOWNTREND 🔴"
                    trend_color = "#F87171"
                elif current_price < sma_20 and sma_20 < sma_50:
                    trend_status = "DOWNTREND 🔴"
                    trend_color = "#F87171"
                else:
                    trend_status = "SIDEWAYS ⚪"
                    trend_color = "#94A3B8"
                
                st.markdown(f"""
                <div class="dashboard-card">
                    <div class="card-title">📊 Trend Analysis</div>
                    <div style="text-align: center; margin: 1rem 0;">
                        <div style="font-size: 1.3rem; font-weight: 700; color: {trend_color};">
                            {trend_status}
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span>SMA 20:</span>
                        <span style="font-weight: 600;">{sma_20:.5f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span>SMA 50:</span>
                        <span style="font-weight: 600;">{sma_50:.5f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Distance 20-50:</span>
                        <span style="font-weight: 600; color: {'#34D399' if sma_20 > sma_50 else '#F87171'};">
                            {(sma_20 - sma_50):.5f}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col_tech2:
            # Momentum Indicators
            st.markdown(f"""
            <div class="dashboard-card">
                <div class="card-title">⚡ Momentum Indicators</div>
                
                <div style="margin-bottom: 1.5rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                        <span>RSI (14):</span>
                        <span style="font-weight: 600; color: {'#F87171' if 'RSI' in df.columns and df['RSI'].iloc[-1] > 70 else '#34D399' if 'RSI' in df.columns and df['RSI'].iloc[-1] < 30 else '#F1F5F9'};">
                            {df['RSI'].iloc[-1] if 'RSI' in df.columns else 'N/A':.1f}
                        </span>
                    </div>
                    <div style="height: 8px; background: var(--border-color); border-radius: 4px; overflow: hidden; margin-bottom: 1rem;">
                        <div style="height: 100%; width: {df['RSI'].iloc[-1] if 'RSI' in df.columns else 50}%; background: {'#F87171' if 'RSI' in df.columns and df['RSI'].iloc[-1] > 70 else '#34D399' if 'RSI' in df.columns and df['RSI'].iloc[-1] < 30 else '#60A5FA'}; border-radius: 4px;"></div>
                    </div>
                    
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span>MACD Line:</span>
                        <span style="font-weight: 600; color: {'#34D399' if 'MACD' in df.columns and df['MACD'].iloc[-1] > 0 else '#F87171'};">
                            {df['MACD'].iloc[-1] if 'MACD' in df.columns else 0:.4f}
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span>MACD Signal:</span>
                        <span style="font-weight: 600;">
                            {df['MACD_signal'].iloc[-1] if 'MACD_signal' in df.columns else 0:.4f}
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span>MACD Histogram:</span>
                        <span style="font-weight: 600; color: {'#34D399' if 'MACD_diff' in df.columns and df['MACD_diff'].iloc[-1] > 0 else '#F87171'};">
                            {df['MACD_diff'].iloc[-1] if 'MACD_diff' in df.columns else 0:.4f}
                        </span>
                    </div>
                </div>
                
                <div style="padding: 1rem; background: var(--bg-secondary); border-radius: 8px;">
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">Momentum Signal</div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                        <span>RSI Signal:</span>
                        <span style="color: {'#34D399' if 'RSI' in df.columns and df['RSI'].iloc[-1] < 30 else '#F87171' if 'RSI' in df.columns and df['RSI'].iloc[-1] > 70 else '#FBBF24'}; font-weight: 600;">
                            {'OVERSOLD' if 'RSI' in df.columns and df['RSI'].iloc[-1] < 30 else 'OVERBOUGHT' if 'RSI' in df.columns and df['RSI'].iloc[-1] > 70 else 'NEUTRAL'}
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>MACD Signal:</span>
                        <span style="color: {'#34D399' if 'MACD' in df.columns and df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1] else '#F87171'}; font-weight: 600;">
                            {'BULLISH' if 'MACD' in df.columns and df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1] else 'BEARISH'}
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Volume Analysis
            volume_avg_20 = df['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = df['volume'].iloc[-1] / volume_avg_20 if volume_avg_20 > 0 else 1
            
            st.markdown(f"""
            <div class="dashboard-card">
                <div class="card-title">📊 Volume Analysis</div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span>Current Volume:</span>
                    <span style="font-weight: 600;">{df['volume'].iloc[-1]:,.0f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span>20-Day Average:</span>
                    <span style="font-weight: 600;">{volume_avg_20:,.0f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                    <span>Volume Ratio:</span>
                    <span style="font-weight: 600; color: {'#34D399' if volume_ratio > 1.2 else '#F87171' if volume_ratio < 0.8 else '#F1F5F9'};">
                        x{volume_ratio:.1f}
                    </span>
                </div>
                
                <div style="padding: 1rem; background: var(--bg-secondary); border-radius: 8px;">
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">Volume Signal</div>
                    <div style="color: {'#34D399' if volume_ratio > 1.2 and price_change > 0 else '#F87171' if volume_ratio > 1.2 and price_change < 0 else '#FBBF24'}; font-weight: 600;">
                        {('HIGH VOLUME BULLISH' if price_change > 0 else 'HIGH VOLUME BEARISH') if volume_ratio > 1.2 else 'NORMAL VOLUME'}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Support and Resistance Levels
        st.markdown("---")
        st.markdown("### 🎯 Support & Resistance Levels")
        
        col_level1, col_level2, col_level3 = st.columns(3)
        
        with col_level1:
            # Pivot Points
            high = df['high'].iloc[-1]
            low = df['low'].iloc[-1]
            close = df['close'].iloc[-1]
            
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            
            st.markdown(f"""
            <div class="dashboard-card">
                <div class="card-title">📊 Pivot Points</div>
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border-color);">
                        <span>Resistance 2:</span>
                        <span style="font-weight: 600; color: #F87171;">{r2:.5f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border-color);">
                        <span>Resistance 1:</span>
                        <span style="font-weight: 600; color: #F87171;">{r1:.5f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border-color);">
                        <span>Pivot Point:</span>
                        <span style="font-weight: 600;">{pivot:.5f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border-color);">
                        <span>Support 1:</span>
                        <span style="font-weight: 600; color: #34D399;">{s1:.5f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Support 2:</span>
                        <span style="font-weight: 600; color: #34D399;">{s2:.5f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_level2:
            # Bollinger Bands
            if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
                bb_upper = df['BB_upper'].iloc[-1]
                bb_lower = df['BB_lower'].iloc[-1]
                bb_middle = df['BB_middle'].iloc[-1]
                bb_width = ((bb_upper - bb_lower) / bb_middle) * 100
                
                st.markdown(f"""
                <div class="dashboard-card">
                    <div class="card-title">📏 Bollinger Bands</div>
                    <div style="margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border-color);">
                            <span>Upper Band:</span>
                            <span style="font-weight: 600; color: #F87171;">{bb_upper:.5f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border-color);">
                            <span>Middle Band:</span>
                            <span style="font-weight: 600;">{bb_middle:.5f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border-color);">
                            <span>Lower Band:</span>
                            <span style="font-weight: 600; color: #34D399;">{bb_lower:.5f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <span>Band Width:</span>
                            <span style="font-weight: 600; color: {'#F87171' if bb_width > 5 else '#34D399' if bb_width < 2 else '#F1F5F9'};">
                                {bb_width:.2f}%
                            </span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span>Position:</span>
                            <span style="font-weight: 600; color: {'#F87171' if current_price > bb_upper * 0.98 else '#34D399' if current_price < bb_lower * 1.02 else '#FBBF24'};">
                                {'NEAR UPPER' if current_price > bb_upper * 0.98 else 'NEAR LOWER' if current_price < bb_lower * 1.02 else 'MIDDLE'}
                            </span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col_level3:
            # Moving Averages
            st.markdown(f"""
            <div class="dashboard-card">
                <div class="card-title">📈 Moving Averages</div>
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border-color);">
                        <span>EMA 9:</span>
                        <span style="font-weight: 600; color: {'#34D399' if current_price > df['EMA_9'].iloc[-1] else '#F87171'};">
                            {df['EMA_9'].iloc[-1] if 'EMA_9' in df.columns else 'N/A':.5f}
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border-color);">
                        <span>SMA 20:</span>
                        <span style="font-weight: 600; color: {'#34D399' if current_price > df['SMA_20'].iloc[-1] else '#F87171'};">
                            {df['SMA_20'].iloc[-1] if 'SMA_20' in df.columns else 'N/A':.5f}
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border-color);">
                        <span>SMA 50:</span>
                        <span style="font-weight: 600; color: {'#34D399' if current_price > df['SMA_50'].iloc[-1] else '#F87171'};">
                            {df['SMA_50'].iloc[-1] if 'SMA_50' in df.columns else 'N/A':.5f}
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>SMA 200:</span>
                        <span style="font-weight: 600; color: {'#34D399' if current_price > df['SMA_200'].iloc[-1] else '#F87171'};">
                            {df['SMA_200'].iloc[-1] if 'SMA_200' in df.columns else 'N/A':.5f}
                        </span>
                    </div>
                </div>
                
                <div style="padding: 1rem; background: var(--bg-secondary); border-radius: 8px;">
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">MA Alignment</div>
                    <div style="color: {'#34D399' if 'EMA_9' in df.columns and 'SMA_20' in df.columns and 'SMA_50' in df.columns and df['EMA_9'].iloc[-1] > df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] else '#F87171' if df['EMA_9'].iloc[-1] < df['SMA_20'].iloc[-1] < df['SMA_50'].iloc[-1] else '#FBBF24'}; font-weight: 600;">
                        {('BULLISH ALIGNMENT' if 'EMA_9' in df.columns and 'SMA_20' in df.columns and 'SMA_50' in df.columns and df['EMA_9'].iloc[-1] > df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] else 'BEARISH ALIGNMENT' if df['EMA_9'].iloc[-1] < df['SMA_20'].iloc[-1] < df['SMA_50'].iloc[-1] else 'MIXED ALIGNMENT')}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Section 4: Detailed technical indicators
        st.markdown("---")
        st.markdown("### 📋 Detailed Technical Indicators")
        
        if st.session_state.show_rsi or st.session_state.show_macd:
            # Create chart for technical indicators
            indicator_fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('RSI (14)', 'MACD'),
                row_heights=[0.5, 0.5]
            )
            
            if st.session_state.show_rsi and 'RSI' in df.columns:
                # RSI
                indicator_fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['RSI'],
                        name='RSI',
                        line=dict(color='#8B5CF6', width=2)
                    ),
                    row=1, col=1
                )
                
                # RSI zones
                indicator_fig.add_hrect(
                    y0=70, y1=100,
                    fillcolor="rgba(248, 113, 113, 0.2)",
                    line_width=0,
                    row=1, col=1
                )
                
                indicator_fig.add_hrect(
                    y0=0, y1=30,
                    fillcolor="rgba(52, 211, 153, 0.2)",
                    line_width=0,
                    row=1, col=1
                )
                
                indicator_fig.add_hline(
                    y=70,
                    line_dash="dash",
                    line_color="#F87171",
                    line_width=1,
                    opacity=0.7,
                    row=1, col=1
                )
                
                indicator_fig.add_hline(
                    y=30,
                    line_dash="dash",
                    line_color="#34D399",
                    line_width=1,
                    opacity=0.7,
                    row=1, col=1
                )
            
            if st.session_state.show_macd and 'MACD' in df.columns:
                # MACD
                indicator_fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['MACD'],
                        name='MACD',
                        line=dict(color='#60A5FA', width=2)
                    ),
                    row=2, col=1
                )
                
                indicator_fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['MACD_signal'],
                        name='Signal',
                        line=dict(color='#F87171', width=2)
                    ),
                    row=2, col=1
                )
                
                # MACD histogram
                colors_macd = ['#34D399' if val > 0 else '#F87171' for val in df['MACD_diff']]
                indicator_fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['MACD_diff'],
                        name='Histogram',
                        marker_color=colors_macd,
                        opacity=0.6
                    ),
                    row=2, col=1
                )
            
            indicator_fig.update_layout(
                height=400,
                showlegend=True,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#F1F5F9'
            )
            
            st.plotly_chart(indicator_fig, use_container_width=True)
        
    else:
        # Error state
        st.error("❌ Unable to load data.")
        st.info("Please check your internet connection or try another pair.")

with tab3:
    # Section 5: Correlation Analysis
    st.markdown("### 🔗 Correlation Analysis")
    
    # Selection of pairs for correlation analysis - maintenant jusqu'à 12 paires
    st.markdown("#### Pair Selection for Analysis")
    
    selected_corr_pairs = st.multiselect(
        "Choose 2 to 12 Forex pairs for correlation analysis:",
        list(FOREX_PAIRS.keys()),
        default=st.session_state.selected_corr_pairs[:5],
        format_func=lambda x: FOREX_PAIRS[x],
        max_selections=12
    )
    
    if selected_corr_pairs != st.session_state.selected_corr_pairs:
        st.session_state.selected_corr_pairs = selected_corr_pairs
    
    if len(selected_corr_pairs) >= 2:
        # Display 1-year pair curves chart
        st.markdown("---")
        st.markdown("### 📈 1-Year Pair Price Comparison")
        
        pair_curves_chart = create_1year_pair_curves_chart(selected_corr_pairs)
        
        if pair_curves_chart:
            st.plotly_chart(pair_curves_chart, use_container_width=True)
            
            # Performance summary
            st.markdown("#### 📊 1-Year Performance Summary")
            
            # Get data for summary
            summary_data = []
            for pair in selected_corr_pairs:
                data = get_1year_data(pair)
                if not data.empty and len(data) > 50:
                    pair_name = FOREX_PAIRS.get(pair, pair)
                    first_price = data['close'].iloc[0]
                    last_price = data['close'].iloc[-1]
                    total_return = ((last_price - first_price) / first_price) * 100
                    max_price = data['close'].max()
                    min_price = data['close'].min()
                    max_return = ((max_price - first_price) / first_price) * 100
                    min_return = ((min_price - first_price) / first_price) * 100
                    
                    summary_data.append({
                        'Pair': pair_name,
                        'Start': f'{first_price:.5f}',
                        'End': f'{last_price:.5f}',
                        'Return': total_return,
                        'Max Return': max_return,
                        'Min Return': min_return
                    })
            
            # Display summary in columns
            if summary_data:
                # Trier par performance
                summary_data.sort(key=lambda x: x['Return'], reverse=True)
                
                # Afficher les meilleures et moins bonnes performances
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                
                with col_sum1:
                    best_pair = summary_data[0]
                    st.markdown(f"""
                    <div class="dashboard-card" style="padding: 1rem; text-align: center;">
                        <div style="font-weight: 600; margin-bottom: 0.5rem; font-size: 0.9rem;">
                            🥇 Best Performer
                        </div>
                        <div style="font-weight: 700; margin-bottom: 0.3rem;">
                            {best_pair['Pair']}
                        </div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #34D399;">
                            {best_pair['Return']:+.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_sum2:
                    # Trouver la performance moyenne
                    avg_return = np.mean([d['Return'] for d in summary_data])
                    st.markdown(f"""
                    <div class="dashboard-card" style="padding: 1rem; text-align: center;">
                        <div style="font-weight: 600; margin-bottom: 0.5rem; font-size: 0.9rem;">
                            📊 Average Return
                        </div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: {'#34D399' if avg_return > 0 else '#F87171'};">
                            {avg_return:+.1f}%
                        </div>
                        <div style="font-size: 0.8rem; color: var(--text-secondary);">
                            {len(summary_data)} pairs
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_sum3:
                    worst_pair = summary_data[-1]
                    st.markdown(f"""
                    <div class="dashboard-card" style="padding: 1rem; text-align: center;">
                        <div style="font-weight: 600; margin-bottom: 0.5rem; font-size: 0.9rem;">
                            🥈 Worst Performer
                        </div>
                        <div style="font-weight: 700; margin-bottom: 0.3rem;">
                            {worst_pair['Pair']}
                        </div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #F87171;">
                            {worst_pair['Return']:+.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("Unable to create 1-year pair comparison chart.")
        
        # Current correlation matrix
        st.markdown("---")
        st.markdown("### 📊 Current Correlation Matrix")
        
        correlation_matrix, close_prices = create_correlation_matrix(
            selected_corr_pairs,
            '1d',  # Utiliser daily pour la corrélation sur une période plus longue
            periods=100
        )
        
        if not correlation_matrix.empty:
            col_corr1, col_corr2 = st.columns([3, 2])
            
            with col_corr1:
                # Correlation heatmap
                fig_corr_heat = px.imshow(
                    correlation_matrix,
                    text_auto='.2f',
                    aspect='auto',
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1,
                    title=f'Correlation Matrix ({len(selected_corr_pairs)} Pairs)'
                )
                
                fig_corr_heat.update_layout(
                    height=500 if len(selected_corr_pairs) > 6 else 400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#F1F5F9'
                )
                
                st.plotly_chart(fig_corr_heat, use_container_width=True)
            
            with col_corr2:
                # Correlation information
                st.markdown("#### 📈 Key Correlations")
                
                # Find extreme correlations
                correlations = []
                for i in range(len(correlation_matrix)):
                    for j in range(i+1, len(correlation_matrix)):
                        pair1 = correlation_matrix.columns[i]
                        pair2 = correlation_matrix.columns[j]
                        corr = correlation_matrix.iloc[i, j]
                        correlations.append((pair1, pair2, corr))
                
                # Sort by absolute value
                correlations.sort(key=lambda x: abs(x[2]), reverse=True)
                
                # Afficher les 6 corrélations les plus fortes
                for pair1, pair2, corr in correlations[:6]:
                    color = "#34D399" if corr > 0 else "#F87171"
                    strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.3 else "Weak"
                    
                    st.markdown(f"""
                    <div class="dashboard-card" style="margin-bottom: 1rem; padding: 1rem;">
                        <div style="font-weight: 600; margin-bottom: 0.3rem; font-size: 0.9rem;">
                            {pair1} ↔ {pair2}
                        </div>
                        <div style="color: {color}; font-weight: 700; font-size: 1.5rem;">
                            {corr:.3f}
                        </div>
                        <div style="font-size: 0.8rem; color: var(--text-secondary);">
                            {strength} {'positive' if corr > 0 else 'negative'} correlation
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
    else:
        st.info("Please select at least 2 pairs for correlation analysis.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--text-secondary); font-size: 0.9rem; padding: 1.5rem;">
    <p>💱 <strong>Forex Pro Dashboard Premium</strong> • Multi-pairs • Correlation • Technical Analysis • Dark Theme</p>
    <p style="font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.7;">
        Version 3.2 • Real-time Data • For Educational Purposes Only
    </p>
</div>
""", unsafe_allow_html=True)