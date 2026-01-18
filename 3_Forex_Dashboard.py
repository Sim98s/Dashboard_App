# dashboard_forex_market_analytics.py
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
    page_title="Forex Market Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with dark theme
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
        height: 100%;
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
    
    /* Nouveaux styles pour Technical Overview */
    .tech-card {
        background: linear-gradient(135deg, var(--bg-card) 0%, #2D3748 100%);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-left: 4px solid var(--accent-color);
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .tech-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    
    .tech-title {
        font-size: 1rem;
        color: var(--accent-color);
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(96, 165, 250, 0.3);
    }
    
    .tech-value {
        font-size: 1.3rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    
    .tech-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-bottom: 0.2rem;
        font-weight: 500;
    }
    
    .tech-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .tech-row:last-child {
        border-bottom: none;
    }
    
    .signal-box {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 8px;
        padding: 0.8rem;
        margin-top: 0.8rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .signal-title {
        font-size: 0.9rem;
        color: var(--accent-color);
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    
    /* Styles pour la barre latérale */
    .sidebar-card {
        background: linear-gradient(135deg, var(--bg-card) 0%, #2D3748 100%);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
    }
    
    .sidebar-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        border-color: var(--accent-color);
    }
    
    .sidebar-title {
        font-size: 1.1rem;
        color: var(--accent-color);
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(96, 165, 250, 0.3);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Style pour les selectbox */
    div[data-baseweb="select"] > div {
        background-color: var(--bg-secondary) !important;
        border-color: var(--border-color) !important;
        color: var(--text-primary) !important;
    }
    
    div[data-baseweb="select"] svg {
        fill: var(--accent-color) !important;
    }
    
    /* Style pour les checkboxes */
    .stCheckbox > label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }
    
    /* Badge de compteur */
    .counter-badge {
        background: var(--accent-color);
        color: white;
        font-size: 0.75rem;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        margin-left: 0.5rem;
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
    
    /* Style pour le multiselect de corrélation */
    .stMultiSelect [data-baseweb="select"] > div {
        background-color: var(--bg-secondary) !important;
        border-color: var(--border-color) !important;
        color: var(--text-primary) !important;
    }
    
    .stMultiSelect [data-baseweb="tag"] {
        background-color: rgba(96, 165, 250, 0.2) !important;
        color: var(--text-primary) !important;
        border-color: var(--accent-color) !important;
    }
    
    .stMultiSelect [data-baseweb="tag"] svg {
        fill: var(--accent-color) !important;
    }
    
    /* Style pour les mini cartes Multi-Pair */
    .mini-pair-card {
        background: var(--bg-card);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .mini-pair-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        border-color: var(--accent-color);
    }
    
    .pair-name-mini {
        font-weight: 600;
        color: var(--text-primary);
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    .pair-price-mini {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.3rem;
    }
    
    .pair-change-mini {
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Indicateur de paire sélectionnée */
    .selected-pair-indicator {
        background: rgba(96, 165, 250, 0.1);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border: 2px solid var(--accent-color);
        text-align: center;
    }
    
    .selected-pair-name {
        color: var(--accent-color);
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    
    .selected-pair-ticker {
        color: var(--text-secondary);
        font-size: 0.9rem;
        font-family: monospace;
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
        'multi_pairs': list(FOREX_PAIRS.keys())[:20],  # MODIFIÉ: les 20 premières paires par défaut
        'correlation_period': '12m',
        'selected_corr_pairs': list(FOREX_PAIRS.keys())[:12]  # Par défaut les 12 premières paires
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
        
        for pair in selected_pairs:
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
        
        # Colors for different pairs
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
            
            # Create hover text
            hover_text = (f'{pair_name}<br>'
                         f'Date: %{{x|%Y-%m-%d}}<br>'
                         f'Normalized Price: %{{y:.2f}}<br>'
                         f'Total Return: {total_return:+.2f}%<br>'
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
            title='FX Pair Correlation - 1Y',
            height=550,
            template='plotly_dark',
            hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#F1F5F9',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(30, 41, 59, 0.8)',
                bordercolor='#334155',
                borderwidth=1
            ),
            margin=dict(b=100),
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
        
        if len(returns) > 5:
            correlation_matrix = returns.corr()
            return correlation_matrix, close_prices
        else:
            return pd.DataFrame(), close_prices
    
    return pd.DataFrame(), pd.DataFrame()

# Function to create multi-pair view - MODIFIÉE POUR 20 PAIRES
def create_multi_pair_view(selected_pairs, timeframe='1h'):
    """Create compact view for multiple pairs - Support jusqu'à 20 paires"""
    st.markdown("### Multi-Pair View")
    
    # Créer des rangées de 5 colonnes pour afficher 20 paires maximum
    num_pairs = min(len(selected_pairs), 20)  # Limite à 20 paires
    num_rows = (num_pairs + 4) // 5  # 5 paires par ligne
    
    for row in range(num_rows):
        cols = st.columns(5)
        start_idx = row * 5
        end_idx = min(start_idx + 5, num_pairs)
        
        for col_idx, pair in enumerate(selected_pairs[start_idx:end_idx]):
            with cols[col_idx]:
                data = get_forex_data(pair, timeframe, periods=50)
                
                if not data.empty:
                    current_price = data['close'].iloc[-1]
                    prev_price = data['close'].iloc[-2] if len(data) > 1 else current_price
                    price_change = ((current_price - prev_price) / prev_price) * 100
                    
                    # Compact card
                    color = '#34D399' if price_change > 0 else '#F87171'
                    sign = '+' if price_change > 0 else ''
                    
                    st.markdown(f"""
                    <div class="mini-pair-card">
                        <div class="pair-name-mini">
                            {FOREX_PAIRS.get(pair, pair)}
                        </div>
                        <div class="pair-price-mini">
                            {current_price:.5f}
                        </div>
                        <div class="pair-change-mini" style="color: {color};">
                            {sign}{price_change:.2f}%
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
    
    # Afficher le compteur de paires
    st.markdown(f"""
    <div style="text-align: center; margin-top: 1rem; padding: 1rem; background: var(--bg-card); border-radius: 8px; border: 1px solid var(--border-color);">
        <div style="color: var(--text-secondary); font-size: 0.9rem;">Displaying</div>
        <div style="color: var(--accent-color); font-size: 1.8rem; font-weight: 700;">{num_pairs}</div>
        <div style="color: var(--text-secondary); font-size: 0.8rem;">/20 Forex pairs</div>
    </div>
    """, unsafe_allow_html=True)

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
        
        for pair in selected_pairs[:20]:  # Limite à 20 paires pour la lisibilité
            data = get_1year_performance_data(pair)
            if not data.empty and 'normalized' in data.columns:
                performance_data[pair] = data
                available_pairs.append(pair)
        
        if len(available_pairs) < 2:
            st.warning("Insufficient data for performance comparison.")
            return None
        
        # Create the chart
        fig = go.Figure()
        
        # Colors for different pairs (ajouté plus de couleurs pour 20 paires)
        colors = ['#60A5FA', '#34D399', '#F87171', '#FBBF24', '#8B5CF6', '#EC4899', 
                  '#10B981', '#F59E0B', '#6366F1', '#EF4444', '#14B8A6', '#F97316',
                  '#3B82F6', '#10B981', '#8B5CF6', '#F59E0B', '#EF4444', '#84CC16',
                  '#F97316', '#6366F1']
        
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
                    line=dict(color=colors[idx % len(colors)], width=1.8 if len(available_pairs) > 10 else 2),
                    mode='lines',
                    hovertemplate=f'{pair_name}<br>Date: %{{x|%Y-%m-%d}}<br>Performance: %{{y:.1f}}%<extra></extra>'
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f'FX Pair Performance - 1Y ({len(available_pairs)} pairs)',
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
                x=1,
                font=dict(size=10 if len(available_pairs) > 10 else 12)
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
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating performance chart: {str(e)}")
        return None

# Sidebar - MODIFIÉE POUR 20 PAIRES
with st.sidebar:
    # En-tête de la sidebar
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: var(--accent-color); margin-bottom: 0.5rem;">📊 Forex Analytics</h2>
        <p style="color: var(--text-secondary); font-size: 0.9rem;">Real-time Market Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Section 1 : Paires à surveiller - TOUTES LES PAIRES VISIBLES (20 paires)
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">📈 Pairs to Monitor</div>', unsafe_allow_html=True)
    
    # Afficher toutes les paires en 2 colonnes
    st.markdown('<div style="margin-bottom: 0.5rem; color: var(--text-primary); font-weight: 600;">Select up to 20 pairs:</div>', unsafe_allow_html=True)
    
    # Créer une grille de checkboxes pour toutes les paires
    all_pairs_items = list(FOREX_PAIRS.items())
    half = len(all_pairs_items) // 2
    if len(all_pairs_items) % 2:
        half += 1
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        for pair_code, pair_name in all_pairs_items[:half]:
            is_selected = pair_code in st.session_state.multi_pairs
            
            if st.checkbox(pair_name, value=is_selected, key=f"multi_{pair_code}"):
                if pair_code not in st.session_state.multi_pairs and len(st.session_state.multi_pairs) < 20:  # MODIFIÉ: 20 au lieu de 12
                    st.session_state.multi_pairs.append(pair_code)
            else:
                if pair_code in st.session_state.multi_pairs:
                    st.session_state.multi_pairs.remove(pair_code)
    
    with col_right:
        for pair_code, pair_name in all_pairs_items[half:]:
            is_selected = pair_code in st.session_state.multi_pairs
            
            if st.checkbox(pair_name, value=is_selected, key=f"multi_r_{pair_code}"):
                if pair_code not in st.session_state.multi_pairs and len(st.session_state.multi_pairs) < 20:  # MODIFIÉ: 20 au lieu de 12
                    st.session_state.multi_pairs.append(pair_code)
            else:
                if pair_code in st.session_state.multi_pairs:
                    st.session_state.multi_pairs.remove(pair_code)
    
    selected_count = len(st.session_state.multi_pairs)
    st.markdown(f'<div style="text-align: center; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border-color); color: var(--text-secondary);">Selected: <span style="color: var(--accent-color); font-weight: 600;">{selected_count}/20</span> pairs</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Section 2 : Paire principale - LISTE DÉROULANTE
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🎯 Main Pair Analysis</div>', unsafe_allow_html=True)
    
    # Liste déroulante pour la paire principale
    st.markdown('<div style="margin-bottom: 0.5rem; color: var(--text-primary); font-weight: 600;">Select main pair:</div>', unsafe_allow_html=True)
    
    # Options pour la liste déroulante
    pair_options = [f"{name} ({code})" for code, name in FOREX_PAIRS.items()]
    
    # Trouver l'index de la paire actuellement sélectionnée
    current_pair_display = f"{FOREX_PAIRS[st.session_state.selected_pair]} ({st.session_state.selected_pair})"
    selected_index = pair_options.index(current_pair_display)
    
    # Widget selectbox
    selected_option = st.selectbox(
        "Choose a pair:",
        options=pair_options,
        index=selected_index,
        label_visibility="collapsed"
    )
    
    # Extraire le code de la paire sélectionnée
    selected_pair_code = selected_option.split(" (")[1].replace(")", "")
    st.session_state.selected_pair = selected_pair_code
    
    # Afficher la paire sélectionnée
    selected_pair_name = FOREX_PAIRS[selected_pair_code]
    st.markdown(f"""
    <div class="selected-pair-indicator">
        <div class="selected-pair-name">{selected_pair_name}</div>
        <div class="selected-pair-ticker">{selected_pair_code}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Section 3 : Timeframe
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">⏰ Timeframe</div>', unsafe_allow_html=True)
    
    timeframe_options = {
        '1m': '1 Minute',
        '5m': '5 Minutes',
        '15m': '15 Minutes',
        '30m': '30 Minutes',
        '1h': '1 Hour',
        '4h': '4 Hours',
        '1d': 'Daily'
    }
    
    # Boutons radio stylés
    selected_tf = st.radio(
        "Select interval:",
        options=list(timeframe_options.keys()),
        format_func=lambda x: timeframe_options[x],
        index=4,
        label_visibility="collapsed"
    )
    st.session_state.timeframe = selected_tf
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Section 4 : Indicateurs techniques
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">📊 Technical Indicators</div>', unsafe_allow_html=True)
    
    # Indicateurs principaux
    st.markdown('<div style="margin-bottom: 0.5rem; color: var(--text-primary); font-weight: 600;">Trend Indicators</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.show_sma = st.checkbox("SMA", st.session_state.show_sma)
        st.session_state.show_ema = st.checkbox("EMA", st.session_state.show_ema)
        st.session_state.show_bollinger = st.checkbox("Bollinger", st.session_state.show_bollinger)
    
    with col2:
        st.session_state.show_rsi = st.checkbox("RSI", st.session_state.show_rsi)
        st.session_state.show_macd = st.checkbox("MACD", st.session_state.show_macd)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main interface
st.markdown('<h1 class="main-header">Forex Market Analytics</h1>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Multi-Pair View", "Technical Analysis", "Correlation Analysis"])

with tab1:
    # Section 1: Multi-Pair View - JUSQU'À 20 PAIRES
    if st.session_state.multi_pairs:
        create_multi_pair_view(st.session_state.multi_pairs, st.session_state.timeframe)
        
        # Add 1-Year Performance Chart
        st.markdown("---")
        st.markdown("### FX Pair Performance - 1Y")
        
        perf_chart = create_1year_performance_chart(st.session_state.multi_pairs)
        if perf_chart:
            st.plotly_chart(perf_chart, use_container_width=True)
        else:
            st.info("Loading 1-year performance data...")
    else:
        st.info("Please select pairs in the sidebar to display the multi-pair view.")

with tab2:
    # Section 2: Technical analysis of main pair
    # Afficher la paire sélectionnée en haut
    selected_pair_name = FOREX_PAIRS.get(st.session_state.selected_pair, st.session_state.selected_pair)
    selected_pair_code = st.session_state.selected_pair
    
    st.markdown(f"""
    <div style="background: var(--bg-card); border-radius: 10px; padding: 1.5rem; margin-bottom: 2rem; border: 1px solid var(--border-color);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="color: var(--accent-color); margin-bottom: 0.5rem;">{selected_pair_name}</h2>
                <div style="color: var(--text-secondary); font-family: monospace; font-size: 1.1rem;">{selected_pair_code}</div>
            </div>
            <div style="text-align: right;">
                <div style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 0.3rem;">Timeframe</div>
                <div style="color: var(--accent-color); font-weight: 700; font-size: 1.2rem;">
                    {st.session_state.timeframe.upper()}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    data = get_forex_data(st.session_state.selected_pair, st.session_state.timeframe, st.session_state.periods)
    
    if not data.empty and len(data) > 10:
        # Calculate indicators
        df = data.copy()
        
        # Calculate indicators conditionally
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
            price_change_class = 'positive-change' if price_change > 0 else 'negative-change'
            price_change_sign = '+' if price_change > 0 else ''
            st.markdown(f"""
            <div class="dashboard-card">
                <div class="card-title">Current Price</div>
                <div class="metric-value">
                    {current_price:.5f}
                </div>
                <div class="{price_change_class}">
                    {price_change_sign}{price_change:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            daily_range = df['high'].iloc[-1] - df['low'].iloc[-1]
            avg_range = (df['high'] - df['low']).mean()
            range_change = ((daily_range - avg_range) / avg_range) * 100 if avg_range > 0 else 0
            range_change_class = 'positive-change' if range_change > 0 else 'negative-change'
            range_change_sign = '+' if range_change > 0 else ''
            
            st.markdown(f"""
            <div class="dashboard-card">
                <div class="card-title">Daily Range</div>
                <div class="metric-value">
                    {daily_range:.5f}
                </div>
                <div class="{range_change_class}">
                    {range_change_sign}{range_change:.2f}% vs avg
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Corriger le problème de volume
            volume = float(df['volume'].iloc[-1]) if not pd.isna(df['volume'].iloc[-1]) else 0
            avg_volume = float(df['volume'].mean()) if len(df) > 0 else 0
            volume_change = ((volume - avg_volume) / avg_volume) * 100 if avg_volume > 0 else 0
            volume_change_class = 'positive-change' if volume_change > 0 else 'negative-change'
            volume_change_sign = '+' if volume_change > 0 else ''
            
            st.markdown(f"""
            <div class="dashboard-card">
                <div class="card-title">Volume</div>
                <div class="metric-value">
                    {volume:,.0f}
                </div>
                <div class="{volume_change_class}">
                    {volume_change_sign}{volume_change:.2f}% vs avg
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            if len(df) > 1:
                daily_returns = df['close'].pct_change().dropna()
                volatility = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 0 else 0
            else:
                volatility = 0
                
            st.markdown(f"""
            <div class="dashboard-card">
                <div class="card-title">Annual Volatility</div>
                <div class="metric-value">
                    {volatility:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            if st.session_state.show_rsi and 'RSI' in df.columns:
                rsi = df['RSI'].iloc[-1]
                rsi_signal = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"
                color = "#F87171" if rsi > 70 else "#34D399" if rsi < 30 else "#94A3B8"
                
                st.markdown(f"""
            <div class="dashboard-card">
                <div class="card-title">RSI (14)</div>
                <div class="metric-value" style="color: {color};">
                    {rsi:.1f}
                </div>
                <div style="font-weight: 600; color: {color};">
                    {rsi_signal}
                </div>
            </div>
            """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
            <div class="dashboard-card">
                <div class="card-title">RSI (14)</div>
                <div class="metric-value">
                    N/A
                </div>
                <div style="font-weight: 600; color: #94A3B8;">
                    Enable RSI in sidebar
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Main chart
        st.markdown("### Main Chart")
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{selected_pair_name} - Technical Analysis', 'Volume')
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
        
        # Indicators - only if enabled
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
        
        if st.session_state.show_bollinger and all(col in df.columns for col in ['BB_upper', 'BB_lower']):
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
        
        # Detailed technical indicators - DÉPLACÉ ICI
        if st.session_state.show_rsi or st.session_state.show_macd:
            st.markdown("### Detailed Technical Indicators")
            
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
            
            if st.session_state.show_macd and all(col in df.columns for col in ['MACD', 'MACD_signal', 'MACD_diff']):
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
        
        # Technical Overview - VERSION CORRIGÉE
        st.markdown("---")
        st.markdown("### Technical Overview")
        
        # Première rangée : 4 cartes
        col_tech1, col_tech2, col_tech3, col_tech4 = st.columns(4)
        
        with col_tech1:
            # Price & Trend Analysis Card
            st.markdown('<div class="tech-card">', unsafe_allow_html=True)
            st.markdown('<div class="tech-title">Price & Trend</div>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="tech-row">
                <span class="tech-label">Price</span>
                <span class="tech-value">{current_price:.5f}</span>
            </div>
            <div class="tech-row">
                <span class="tech-label">24h Change</span>
                <span style="color: {'#34D399' if price_change > 0 else '#F87171'}; font-weight: 600;">
                    {'+' if price_change > 0 else ''}{price_change:.2f}%
                </span>
            </div>
            <div class="tech-row">
                <span class="tech-label">Range</span>
                <span class="tech-value">{daily_range:.5f}</span>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.show_sma and all(col in df.columns for col in ['SMA_20', 'SMA_50']):
                sma_20 = df['SMA_20'].iloc[-1]
                sma_50 = df['SMA_50'].iloc[-1]
                
                # Détermination de la tendance
                if current_price > sma_20 > sma_50:
                    trend_status = "STRONG UPTREND"
                    trend_color = "#34D399"
                elif current_price > sma_20 and sma_20 < sma_50:
                    trend_status = "WEAK UPTREND"
                    trend_color = "#34D399"
                elif current_price < sma_20 < sma_50:
                    trend_status = "STRONG DOWNTREND"
                    trend_color = "#F87171"
                elif current_price < sma_20 and sma_20 > sma_50:
                    trend_status = "WEAK DOWNTREND"
                    trend_color = "#F87171"
                else:
                    trend_status = "SIDEWAYS"
                    trend_color = "#94A3B8"
                
                st.markdown(f"""
                <div class="signal-box">
                    <div class="signal-title">Trend Signal</div>
                    <div style="color: {trend_color}; font-weight: 600; text-align: center;">
                        {trend_status}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_tech2:
            # Momentum Indicators Card
            st.markdown('<div class="tech-card">', unsafe_allow_html=True)
            st.markdown('<div class="tech-title">Momentum</div>', unsafe_allow_html=True)
            
            if st.session_state.show_rsi and 'RSI' in df.columns:
                rsi_value = df['RSI'].iloc[-1]
                rsi_color = "#F87171" if rsi_value > 70 else "#34D399" if rsi_value < 30 else "#F1F5F9"
                
                st.markdown(f"""
                <div class="tech-row">
                    <span class="tech-label">RSI (14)</span>
                    <span style="color: {rsi_color}; font-weight: 700;">
                        {rsi_value:.1f}
                    </span>
                </div>
                <div style="height: 6px; background: var(--border-color); border-radius: 3px; overflow: hidden; margin: 0.3rem 0;">
                    <div style="height: 100%; width: {min(rsi_value, 100)}%; background: {rsi_color}; border-radius: 3px;"></div>
                </div>
                """, unsafe_allow_html=True)
            
            if st.session_state.show_macd and all(col in df.columns for col in ['MACD', 'MACD_signal']):
                macd_value = df['MACD'].iloc[-1]
                macd_signal = df['MACD_signal'].iloc[-1]
                macd_bullish = macd_value > macd_signal
                
                st.markdown(f"""
                <div class="tech-row">
                    <span class="tech-label">MACD</span>
                    <span style="color: {'#34D399' if macd_bullish else '#F87171'}; font-weight: 600;">
                        {macd_value:.4f}
                    </span>
                </div>
                <div class="tech-row">
                    <span class="tech-label">Signal</span>
                    <span style="font-weight: 600;">{macd_signal:.4f}</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Momentum Signal
            momentum_signal = ""
            momentum_color = "#FBBF24"
            
            if st.session_state.show_rsi and 'RSI' in df.columns:
                rsi_value = df['RSI'].iloc[-1]
                if rsi_value < 30:
                    momentum_signal = "OVERSOLD"
                    momentum_color = "#34D399"
                elif rsi_value > 70:
                    momentum_signal = "OVERBOUGHT"
                    momentum_color = "#F87171"
                else:
                    momentum_signal = "NEUTRAL"
                    momentum_color = "#FBBF24"
            
            if st.session_state.show_macd and all(col in df.columns for col in ['MACD', 'MACD_signal']):
                macd_value = df['MACD'].iloc[-1]
                macd_signal = df['MACD_signal'].iloc[-1]
                if macd_value > macd_signal and momentum_signal != "OVERBOUGHT":
                    momentum_signal = "BULLISH"
                    momentum_color = "#34D399"
                elif macd_value < macd_signal and momentum_signal != "OVERSOLD":
                    momentum_signal = "BEARISH"
                    momentum_color = "#F87171"
            
            st.markdown(f"""
            <div class="signal-box">
                <div class="signal-title">Momentum Signal</div>
                <div style="color: {momentum_color}; font-weight: 600; text-align: center;">
                    {momentum_signal}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_tech3:
            # Volume & Volatility Card
            st.markdown('<div class="tech-card">', unsafe_allow_html=True)
            st.markdown('<div class="tech-title">Volume & Volatility</div>', unsafe_allow_html=True)
            
            # Volume
            volume = float(df['volume'].iloc[-1]) if not pd.isna(df['volume'].iloc[-1]) else 0
            if len(df) >= 20:
                volume_avg_20 = float(df['volume'].rolling(20).mean().iloc[-1]) if not pd.isna(df['volume'].rolling(20).mean().iloc[-1]) else 0
            else:
                volume_avg_20 = float(df['volume'].mean()) if len(df) > 0 else 0
            
            volume_ratio = volume / volume_avg_20 if volume_avg_20 > 0 else 1
            
            st.markdown(f"""
            <div class="tech-row">
                <span class="tech-label">Volume</span>
                <span class="tech-value">{volume:,.0f}</span>
            </div>
            <div class="tech-row">
                <span class="tech-label">20D Avg</span>
                <span class="tech-value">{volume_avg_20:,.0f}</span>
            </div>
            <div class="tech-row">
                <span class="tech-label">Ratio</span>
                <span style="color: {'#34D399' if volume_ratio > 1.2 else '#F87171' if volume_ratio < 0.8 else '#F1F5F9'}; font-weight: 600;">
                    x{volume_ratio:.1f}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Volatility
            if len(df) > 1:
                daily_returns = df['close'].pct_change().dropna()
                volatility = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 0 else 0
            else:
                volatility = 0
            
            # ATR
            try:
                if all(col in df.columns for col in ['high', 'low', 'close']):
                    atr_value = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14).iloc[-1]
                    atr_display = f"{atr_value:.5f}"
                else:
                    atr_display = "N/A"
            except:
                atr_display = "N/A"
            
            st.markdown(f"""
            <div class="tech-row">
                <span class="tech-label">ATR (14)</span>
                <span class="tech-value">{atr_display}</span>
            </div>
            <div class="tech-row">
                <span class="tech-label">Ann. Vol</span>
                <span class="tech-value">{volatility:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Volume Signal
            if volume_ratio > 1.2 and price_change > 0:
                volume_signal = "HIGH VOL BULLISH"
                volume_signal_color = "#34D399"
            elif volume_ratio > 1.2 and price_change < 0:
                volume_signal = "HIGH VOL BEARISH"
                volume_signal_color = "#F87171"
            elif volume_ratio < 0.8:
                volume_signal = "LOW VOLUME"
                volume_signal_color = "#FBBF24"
            else:
                volume_signal = "NORMAL VOL"
                volume_signal_color = "#F1F5F9"
            
            st.markdown(f"""
            <div class="signal-box">
                <div class="signal-title">Volume Signal</div>
                <div style="color: {volume_signal_color}; font-weight: 600; text-align: center;">
                    {volume_signal}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_tech4:
            # Moving Averages Card
            st.markdown('<div class="tech-card">', unsafe_allow_html=True)
            st.markdown('<div class="tech-title">Moving Averages</div>', unsafe_allow_html=True)
            
            ma_values = []
            ma_labels = []
            
            if st.session_state.show_ema and 'EMA_9' in df.columns:
                ema_9 = df['EMA_9'].iloc[-1]
                ma_values.append(ema_9)
                ma_labels.append("EMA 9")
                st.markdown(f"""
                <div class="tech-row">
                    <span class="tech-label">EMA 9</span>
                    <span style="color: {'#34D399' if current_price > ema_9 else '#F87171'}; font-weight: 600;">
                        {ema_9:.5f}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            if st.session_state.show_sma and 'SMA_20' in df.columns:
                sma_20 = df['SMA_20'].iloc[-1]
                ma_values.append(sma_20)
                ma_labels.append("SMA 20")
                st.markdown(f"""
                <div class="tech-row">
                    <span class="tech-label">SMA 20</span>
                    <span style="color: {'#34D399' if current_price > sma_20 else '#F87171'}; font-weight: 600;">
                        {sma_20:.5f}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            if st.session_state.show_sma and 'SMA_50' in df.columns:
                sma_50 = df['SMA_50'].iloc[-1]
                ma_values.append(sma_50)
                ma_labels.append("SMA 50")
                st.markdown(f"""
                <div class="tech-row">
                    <span class="tech-label">SMA 50</span>
                    <span style="color: {'#34D399' if current_price > sma_50 else '#F87171'}; font-weight: 600;">
                        {sma_50:.5f}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            if st.session_state.show_sma and 'SMA_200' in df.columns:
                sma_200 = df['SMA_200'].iloc[-1]
                ma_values.append(sma_200)
                ma_labels.append("SMA 200")
                st.markdown(f"""
                <div class="tech-row">
                    <span class="tech-label">SMA 200</span>
                    <span style="color: {'#34D399' if current_price > sma_200 else '#F87171'}; font-weight: 600;">
                        {sma_200:.5f}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            # Bollinger Bands
            if st.session_state.show_bollinger and all(col in df.columns for col in ['BB_upper', 'BB_lower', 'BB_middle']):
                bb_upper = df['BB_upper'].iloc[-1]
                bb_lower = df['BB_lower'].iloc[-1]
                bb_middle = df['BB_middle'].iloc[-1]
                
                if current_price > bb_upper * 0.98:
                    bb_position = "NEAR UPPER"
                    bb_position_color = "#F87171"
                elif current_price < bb_lower * 1.02:
                    bb_position = "NEAR LOWER"
                    bb_position_color = "#34D399"
                else:
                    bb_position = "MIDDLE"
                    bb_position_color = "#FBBF24"
                
                st.markdown(f"""
                <div class="signal-box">
                    <div class="signal-title">Bollinger Position</div>
                    <div style="color: {bb_position_color}; font-weight: 600; text-align: center;">
                        {bb_position}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # MA Alignment
            if len(ma_values) >= 3:
                # Check alignment
                if ma_values[0] > ma_values[1] > ma_values[2]:
                    alignment = "BULLISH"
                    alignment_color = "#34D399"
                elif ma_values[0] < ma_values[1] < ma_values[2]:
                    alignment = "BEARISH"
                    alignment_color = "#F87171"
                else:
                    alignment = "MIXED"
                    alignment_color = "#FBBF24"
                
                st.markdown(f"""
                <div class="signal-box" style="margin-top: 0.8rem;">
                    <div class="signal-title">MA Alignment</div>
                    <div style="color: {alignment_color}; font-weight: 600; text-align: center;">
                        {alignment}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        # Error state
        st.error("Unable to load data.")
        st.info("Please check your internet connection or try another pair.")

with tab3:
    # Section 3: Correlation Analysis - SIMPLIFIÉE
    st.markdown("### Correlation Analysis")
    
    # Sélection des paires pour l'analyse de corrélation
    col_corr_top1, col_corr_top2 = st.columns([3, 1])
    
    with col_corr_top1:
        # Multiselect pour TOUTES les 20 paires
        selected_corr_pairs = st.multiselect(
            "Select Forex pairs for correlation analysis:",
            list(FOREX_PAIRS.keys()),
            default=list(FOREX_PAIRS.keys())[:12],  # Par défaut les 12 premières paires
            format_func=lambda x: FOREX_PAIRS[x],
            max_selections=20,  # Permet de sélectionner jusqu'à 20 paires
            label_visibility="collapsed"
        )
    
    with col_corr_top2:
        selected_count = len(selected_corr_pairs) if selected_corr_pairs else 0
        st.markdown(f"""
        <div style="background: var(--bg-card); border-radius: 8px; padding: 1rem; text-align: center; border: 1px solid var(--border-color);">
            <div style="color: var(--text-secondary); font-size: 0.9rem;">Selected Pairs</div>
            <div style="color: var(--accent-color); font-size: 1.8rem; font-weight: 700;">{selected_count}</div>
            <div style="color: var(--text-secondary); font-size: 0.8rem;">/20 maximum</div>
        </div>
        """, unsafe_allow_html=True)
    
    if selected_corr_pairs:
        st.session_state.selected_corr_pairs = selected_corr_pairs
    
    if len(selected_corr_pairs) >= 2:
        # Display 1-year pair curves chart
        st.markdown("---")
        st.markdown("### FX Pair Correlation - 1Y")
        
        pair_curves_chart = create_1year_pair_curves_chart(selected_corr_pairs)
        
        if pair_curves_chart:
            st.plotly_chart(pair_curves_chart, use_container_width=True)
        else:
            st.info("Loading correlation data...")
        
        # Current correlation matrix
        st.markdown("---")
        st.markdown("### FX Pair Correlation Matrix")
        
        correlation_matrix, close_prices = create_correlation_matrix(
            selected_corr_pairs,
            '1d',
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
                    title=f'FX Pair Correlation Matrix ({len(selected_corr_pairs)} Pairs)'
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
                st.markdown("#### Key Correlations")
                
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
                
                # Display the 6 strongest correlations
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
    <p>Forex Market Analytics • Multi-pairs • Correlation • Technical Analysis • Dark Theme</p>
    <p style="font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.7;">
        Version 3.6 • Forex Market Analytics • 20 Pairs Limit
    </p>
</div>
""", unsafe_allow_html=True)