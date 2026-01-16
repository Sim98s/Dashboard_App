import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.linear_model import LinearRegression
import yfinance as yf

class ForexStatisticalArbitrage:
    def __init__(self, pair1, pair2, lookback_period=60, z_score_threshold=2.0):
        """
        Stratégie d'arbitrage statistique pour le Forex
        
        Args:
            pair1 (str): Première paire forex (ex: 'EURUSD=X')
            pair2 (str): Deuxième paire forex (ex: 'GBPUSD=X')
            lookback_period (int): Période de calcul en jours
            z_score_threshold (float): Seuil pour générer les signaux
        """
        self.pair1 = pair1
        self.pair2 = pair2
        self.lookback_period = lookback_period
        self.z_score_threshold = z_score_threshold
        self.hedge_ratio = None
        self.spread_mean = None
        self.spread_std = None
        self.signals = pd.DataFrame()
        
    def fetch_forex_data(self, start_date, end_date):
        """
        Récupère les données forex via Yahoo Finance
        Yahoo Finance utilise:
        - EURUSD=X pour EUR/USD
        - GBPUSD=X pour GBP/USD
        - USDJPY=X pour USD/JPY
        - AUDUSD=X pour AUD/USD
        - USDCAD=X pour USD/CAD
        """
        print(f"Téléchargement des données pour {self.pair1} et {self.pair2}...")
        
        # Téléchargement des données
        data1 = yf.download(self.pair1, start=start_date, end=end_date, progress=False)
        data2 = yf.download(self.pair2, start=start_date, end=end_date, progress=False)
        
        # Utiliser les prix de clôture ajustés
        prices1 = data1['Adj Close'].rename(self.pair1)
        prices2 = data2['Adj Close'].rename(self.pair2)
        
        # Combiner les données
        prices = pd.concat([prices1, prices2], axis=1)
        prices = prices.dropna()
        
        print(f"Données récupérées: {len(prices)} points")
        print(f"Période: {prices.index[0].date()} au {prices.index[-1].date()}")
        
        return prices
    
    def calculate_spread(self, prices):
        """
        Calcule le spread entre les deux paires
        Utilise une régression linéaire pour le hedge ratio
        """
        X = prices[self.pair1].values.reshape(-1, 1)
        y = prices[self.pair2].values
        
        # Calcul du hedge ratio par régression linéaire
        model = LinearRegression()
        model.fit(X, y)
        self.hedge_ratio = model.coef_[0]
        
        # Calcul du spread
        spread = prices[self.pair2] - self.hedge_ratio * prices[self.pair1]
        
        # Normalisation Z-score
        self.spread_mean = spread.mean()
        self.spread_std = spread.std()
        z_score = (spread - self.spread_mean) / self.spread_std
        
        return spread, z_score
    
    def generate_signals(self, prices):
        """
        Génère les signaux de trading basés sur le Z-score
        """
        spread, z_score = self.calculate_spread(prices)
        
        # Initialisation des signaux
        signals = pd.DataFrame(index=prices.index)
        signals['price1'] = prices[self.pair1]
        signals['price2'] = prices[self.pair2]
        signals['spread'] = spread
        signals['z_score'] = z_score
        
        # Génération des signaux
        signals['signal'] = 0
        
        # Signal d'achat: spread est bas (z-score négatif)
        buy_condition = signals['z_score'] < -self.z_score_threshold
        signals.loc[buy_condition, 'signal'] = 1
        
        # Signal de vente: spread est haut (z-score positif)
        sell_condition = signals['z_score'] > self.z_score_threshold
        signals.loc[sell_condition, 'signal'] = -1
        
        # Signal de sortie: retour vers la moyenne
        exit_condition = signals['z_score'].abs() < 0.5
        signals.loc[exit_condition, 'signal'] = 0
        
        # Positions
        signals['position'] = signals['signal'].replace(0, pd.NA).ffill().fillna(0)
        
        self.signals = signals
        return signals
    
    def calculate_pnl(self, prices, signals):
        """
        Calcule le P&L théorique de la stratégie
        """
        # Retours des prix
        returns1 = prices[self.pair1].pct_change()
        returns2 = prices[self.pair2].pct_change()
        
        # Spread des retours
        spread_returns = returns2 - self.hedge_ratio * returns1
        
        # P&L basé sur la position
        pnl = signals['position'].shift(1) * spread_returns
        
        # Cumul du P&L
        cumulative_pnl = (1 + pnl).cumprod()
        
        return pnl, cumulative_pnl
    
    def backtest(self, start_date='2023-01-01', end_date='2024-01-01'):
        """
        Backtest complet de la stratégie
        """
        print("=" * 50)
        print("BACKTEST STRATÉGIE STAT ARB FOREX")
        print(f"Paire 1: {self.pair1}")
        print(f"Paire 2: {self.pair2}")
        print(f"Période: {start_date} au {end_date}")
        print("=" * 50)
        
        # Récupération des données
        prices = self.fetch_forex_data(start_date, end_date)
        
        if len(prices) < self.lookback_period:
            print("Données insuffisantes pour l'analyse")
            return
        
        # Génération des signaux
        signals = self.generate_signals(prices)
        
        # Calcul du P&L
        pnl, cumulative_pnl = self.calculate_pnl(prices, signals)
        
        # Métriques de performance
        total_return = cumulative_pnl.iloc[-1] - 1
        sharpe_ratio = np.sqrt(252) * pnl.mean() / pnl.std() if pnl.std() > 0 else 0
        max_drawdown = (cumulative_pnl / cumulative_pnl.cummax() - 1).min()
        
        print(f"\n--- PERFORMANCE ---")
        print(f"Retour total: {total_return:.2%}")
        print(f"Ratio de Sharpe: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Hedge Ratio: {self.hedge_ratio:.4f}")
        print(f"Nombre de trades: {(signals['signal'].diff() != 0).sum()}")
        
        return signals, cumulative_pnl
    
    def plot_results(self, signals, cumulative_pnl):
        """
        Visualise les résultats
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Graphique 1: Prix normalisés
        norm_price1 = signals['price1'] / signals['price1'].iloc[0]
        norm_price2 = signals['price2'] / signals['price2'].iloc[0]
        
        axes[0].plot(norm_price1.index, norm_price1, label=self.pair1, linewidth=1.5)
        axes[0].plot(norm_price2.index, norm_price2, label=self.pair2, linewidth=1.5)
        axes[0].set_title('Prix Normalisés des Paires Forex')
        axes[0].set_ylabel('Prix Normalisé')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Graphique 2: Spread et Z-score avec signaux
        ax2 = axes[1]
        ax2.plot(signals.index, signals['z_score'], 'b-', linewidth=1.5, label='Z-Score')
        ax2.axhline(y=self.z_score_threshold, color='r', linestyle='--', alpha=0.7, label=f'Seuil ({self.z_score_threshold})')
        ax2.axhline(y=-self.z_score_threshold, color='g', linestyle='--', alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Signaux d'achat
        buy_signals = signals[signals['signal'] == 1]
        ax2.scatter(buy_signals.index, buy_signals['z_score'], color='green', s=100, marker='^', label='Achat')
        
        # Signaux de vente
        sell_signals = signals[signals['signal'] == -1]
        ax2.scatter(sell_signals.index, sell_signals['z_score'], color='red', s=100, marker='v', label='Vente')
        
        ax2.set_title('Z-Score et Signaux de Trading')
        ax2.set_ylabel('Z-Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Graphique 3: P&L Cumulatif
        axes[2].plot(cumulative_pnl.index, cumulative_pnl, 'g-', linewidth=2)
        axes[2].set_title('P&L Cumulatif de la Stratégie')
        axes[2].set_ylabel('P&L Cumulatif')
        axes[2].set_xlabel('Date')
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=1, color='k', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_current_signal(self):
        """
        Obtient le signal actuel en temps quasi-réel
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_period * 2)
        
        prices = self.fetch_forex_data(start_date.strftime('%Y-%m-%d'), 
                                      end_date.strftime('%Y-%m-%d'))
        
        if len(prices) < self.lookback_period:
            return None
        
        signals = self.generate_signals(prices)
        latest_signal = signals.iloc[-1]
        
        return {
            'date': latest_signal.name,
            'pair1_price': latest_signal['price1'],
            'pair2_price': latest_signal['price2'],
            'spread': latest_signal['spread'],
            'z_score': latest_signal['z_score'],
            'signal': latest_signal['signal'],
            'position': latest_signal['position'],
            'hedge_ratio': self.hedge_ratio
        }

# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

def main():
    """
    Exemple d'exécution de la stratégie
    """
    
    # Paires forex corrélées (ex: EUR/USD et GBP/USD)
    # Format Yahoo Finance: XXXYYY=X
    strategy = ForexStatisticalArbitrage(
        pair1='EURUSD=X',      # EUR/USD
        pair2='GBPUSD=X',      # GBP/USD
        lookback_period=60,    # 60 jours pour le calcul
        z_score_threshold=1.5  # Seuil de signal
    )
    
    # Backtest sur l'année 2023
    signals, cumulative_pnl = strategy.backtest(
        start_date='2023-01-01',
        end_date='2024-01-01'
    )
    
    # Affichage des résultats
    strategy.plot_results(signals, cumulative_pnl)
    
    # Signal actuel
    current_signal = strategy.get_current_signal()
    if current_signal:
        print("\n--- SIGNAL ACTUEL ---")
        print(f"Date: {current_signal['date']}")
        print(f"{strategy.pair1}: {current_signal['pair1_price']:.5f}")
        print(f"{strategy.pair2}: {current_signal['pair2_price']:.5f}")
        print(f"Z-Score: {current_signal['z_score']:.2f}")
        print(f"Signal: {current_signal['signal']}")
        print(f"Position: {current_signal['position']}")
        
        # Interprétation du signal
        if current_signal['signal'] == 1:
            print("ACTION: ACHETER la paire2, VENDRE la paire1")
        elif current_signal['signal'] == -1:
            print("ACTION: VENDRE la paire2, ACHETER la paire1")
        else:
            print("ACTION: NEUTRE - Attendre un signal")

# =============================================================================
# AUTRES EXEMPLES DE PAIRES
# =============================================================================

def example_cross_rates():
    """Exemple avec des paires croisées"""
    
    print("\n" + "="*50)
    print("EXEMPLE AVEC PAIRE CROISÉE")
    print("="*50)
    
    # EUR/GBP vs EUR/CHF (paires partageant la même devise de base)
    strategy2 = ForexStatisticalArbitrage(
        pair1='EURGBP=X',      # EUR/GBP
        pair2='EURCHF=X',      # EUR/CHF
        lookback_period=90,
        z_score_threshold=2.0
    )
    
    signals2, pnl2 = strategy2.backtest('2023-01-01', '2024-01-01')
    strategy2.plot_results(signals2, pnl2)

def example_majors():
    """Exemple avec principales paires"""
    
    print("\n" + "="*50)
    print("EXEMPLE USD/CAD vs USD/CHF")
    print("="*50)
    
    strategy3 = ForexStatisticalArbitrage(
        pair1='USDCAD=X',      # USD/CAD
        pair2='USDCHF=X',      # USD/CHF
        lookback_period=75,
        z_score_threshold=1.8
    )
    
    signals3, pnl3 = strategy3.backtest('2023-01-01', '2024-01-01')
    strategy3.plot_results(signals3, pnl3)

if __name__ == "__main__":
    # Exécuter la stratégie principale
    main()
    
    # Démarrer d'autres exemples (décommentez pour exécuter)
    # example_cross_rates()
    # example_majors()