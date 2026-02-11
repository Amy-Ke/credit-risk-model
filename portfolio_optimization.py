"""
Portfolio Optimization - Modern Portfolio Theory
Author: Amy Ke
Description: Implements Markowitz mean-variance optimization to construct
the efficient frontier and identify optimal portfolios across 10 stocks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    def __init__(self, tickers=None, start_date=None, end_date=None,
                 num_portfolios=5000, risk_free_rate=0.05):
        """
        Initialize Portfolio Optimizer
        
        Parameters:
        - tickers: List of stock tickers
        - start_date: Start date for historical data
        - end_date: End date for historical data
        - num_portfolios: Number of random portfolios to simulate
        - risk_free_rate: Annual risk-free rate (default: 5%)
        """
        self.tickers = tickers or [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM',
            'JNJ', 'V', 'PG', 'XOM', 'BRK-B'
        ]
        self.num_portfolios = num_portfolios
        self.risk_free_rate = risk_free_rate
        
        if end_date is None:
            self.end_date = datetime.now()
        else:
            self.end_date = end_date
            
        if start_date is None:
            self.start_date = self.end_date - timedelta(days=365*5)
        else:
            self.start_date = start_date

    def download_data(self):
        """Download historical price data"""
        print("="*60)
        print("PORTFOLIO OPTIMIZATION - MODERN PORTFOLIO THEORY")
        print("="*60)
        print(f"\nDownloading data for {len(self.tickers)} stocks...")
        print(f"Tickers: {', '.join(self.tickers)}")
        
        raw = yf.download(self.tickers, 
                         start=self.start_date,
                         end=self.end_date,
                         auto_adjust=True)['Close']
        
        # Drop any tickers with too many missing values
        self.prices = raw.dropna(axis=1, thresh=int(len(raw)*0.9))
        self.tickers = list(self.prices.columns)
        
        print(f"Downloaded {len(self.prices)} days of data")
        print(f"Successfully loaded: {', '.join(self.tickers)}")
        return self.prices

    def calculate_returns(self):
        """Calculate daily and annual returns"""
        self.daily_returns = self.prices.pct_change().dropna()
        self.annual_returns = self.daily_returns.mean() * 252
        self.cov_matrix = self.daily_returns.cov() * 252
        
        print("\n" + "="*60)
        print("STOCK PERFORMANCE SUMMARY (5-Year Annual)")
        print("="*60)
        for ticker in self.tickers:
            ret = self.annual_returns[ticker]
            vol = np.sqrt(self.cov_matrix.loc[ticker, ticker])
            sharpe = (ret - self.risk_free_rate) / vol
            print(f"{ticker:.<10} Return: {ret:>7.2%} | "
                  f"Volatility: {vol:>7.2%} | "
                  f"Sharpe: {sharpe:>6.2f}")

    def simulate_portfolios(self):
        """Monte Carlo simulation of random portfolios"""
        print(f"\nSimulating {self.num_portfolios:,} random portfolios...")
        
        n = len(self.tickers)
        results = np.zeros((3, self.num_portfolios))
        weights_record = []
        
        for i in range(self.num_portfolios):
            # Random weights
            weights = np.random.random(n)
            weights /= np.sum(weights)
            weights_record.append(weights)
            
            # Portfolio metrics
            port_return = np.sum(weights * self.annual_returns)
            port_vol = np.sqrt(
                np.dot(weights.T, np.dot(self.cov_matrix, weights))
            )
            sharpe = (port_return - self.risk_free_rate) / port_vol
            
            results[0, i] = port_vol
            results[1, i] = port_return
            results[2, i] = sharpe
        
        self.results = results
        self.weights_record = weights_record
        print("Simulation complete!")
        return results

    def find_optimal_portfolios(self):
        """Find key optimal portfolios"""
        # Maximum Sharpe Ratio Portfolio
        max_sharpe_idx = np.argmax(self.results[2])
        self.max_sharpe_weights = self.weights_record[max_sharpe_idx]
        self.max_sharpe_return = self.results[1, max_sharpe_idx]
        self.max_sharpe_vol = self.results[0, max_sharpe_idx]
        self.max_sharpe_ratio = self.results[2, max_sharpe_idx]
        
        # Minimum Volatility Portfolio
        min_vol_idx = np.argmin(self.results[0])
        self.min_vol_weights = self.weights_record[min_vol_idx]
        self.min_vol_return = self.results[1, min_vol_idx]
        self.min_vol_vol = self.results[0, min_vol_idx]
        self.min_vol_sharpe = self.results[2, min_vol_idx]

        print("\n" + "="*60)
        print("OPTIMAL PORTFOLIOS")
        print("="*60)
        
        print("\n🏆 MAXIMUM SHARPE RATIO PORTFOLIO (Best Risk-Adjusted)")
        print(f"   Expected Return:  {self.max_sharpe_return:.2%}")
        print(f"   Volatility:       {self.max_sharpe_vol:.2%}")
        print(f"   Sharpe Ratio:     {self.max_sharpe_ratio:.4f}")
        print("\n   Allocation:")
        for ticker, weight in zip(self.tickers, self.max_sharpe_weights):
            if weight > 0.01:
                print(f"   {ticker:.<10} {weight:.2%}")
        
        print("\n🛡️  MINIMUM VOLATILITY PORTFOLIO (Lowest Risk)")
        print(f"   Expected Return:  {self.min_vol_return:.2%}")
        print(f"   Volatility:       {self.min_vol_vol:.2%}")
        print(f"   Sharpe Ratio:     {self.min_vol_sharpe:.4f}")
        print("\n   Allocation:")
        for ticker, weight in zip(self.tickers, self.min_vol_weights):
            if weight > 0.01:
                print(f"   {ticker:.<10} {weight:.2%}")

    def plot_results(self):
        """Create portfolio optimization visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Efficient Frontier
        ax1 = axes[0, 0]
        scatter = ax1.scatter(
            self.results[0], self.results[1],
            c=self.results[2], cmap='viridis',
            alpha=0.5, s=10
        )
        plt.colorbar(scatter, ax=ax1, label='Sharpe Ratio')
        
        ax1.scatter(
            self.max_sharpe_vol, self.max_sharpe_return,
            marker='*', color='red', s=500,
            label=f'Max Sharpe ({self.max_sharpe_ratio:.2f})',
            zorder=5
        )
        ax1.scatter(
            self.min_vol_vol, self.min_vol_return,
            marker='*', color='blue', s=500,
            label=f'Min Volatility',
            zorder=5
        )
        
        # Plot individual stocks
        for ticker in self.tickers:
            ret = self.annual_returns[ticker]
            vol = np.sqrt(self.cov_matrix.loc[ticker, ticker])
            ax1.scatter(vol, ret, marker='o', s=100, zorder=6)
            ax1.annotate(ticker, (vol, ret),
                        textcoords="offset points",
                        xytext=(5, 5), fontsize=8)
        
        ax1.set_xlabel('Annual Volatility (Risk)', fontsize=11)
        ax1.set_ylabel('Annual Return', fontsize=11)
        ax1.set_title('Efficient Frontier\n(5,000 Random Portfolios)',
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax1.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        
        # Plot 2: Max Sharpe Portfolio Allocation
        ax2 = axes[0, 1]
        filtered_weights = [(t, w) for t, w in 
                           zip(self.tickers, self.max_sharpe_weights)
                           if w > 0.01]
        labels, weights = zip(*filtered_weights)
        colors = cm.Set3(np.linspace(0, 1, len(labels)))
        wedges, texts, autotexts = ax2.pie(
            weights, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90,
            pctdistance=0.85
        )
        ax2.set_title('Max Sharpe Ratio Portfolio\nAllocation',
                     fontsize=13, fontweight='bold')
        
        # Plot 3: Min Volatility Portfolio Allocation
        ax3 = axes[1, 0]
        filtered_weights_mv = [(t, w) for t, w in 
                               zip(self.tickers, self.min_vol_weights)
                               if w > 0.01]
        labels_mv, weights_mv = zip(*filtered_weights_mv)
        colors_mv = cm.Set2(np.linspace(0, 1, len(labels_mv)))
        ax3.pie(
            weights_mv, labels=labels_mv, colors=colors_mv,
            autopct='%1.1f%%', startangle=90,
            pctdistance=0.85
        )
        ax3.set_title('Minimum Volatility Portfolio\nAllocation',
                     fontsize=13, fontweight='bold')
        
        # Plot 4: Individual Stock Risk-Return
        ax4 = axes[1, 1]
        returns = [self.annual_returns[t] for t in self.tickers]
        vols = [np.sqrt(self.cov_matrix.loc[t, t]) 
                for t in self.tickers]
        sharpes = [(r - self.risk_free_rate) / v 
                  for r, v in zip(returns, vols)]
        
        colors_bar = ['green' if s > 0 else 'red' for s in sharpes]
        bars = ax4.bar(self.tickers, sharpes, color=colors_bar, 
                      alpha=0.7, edgecolor='black', linewidth=0.5)
        ax4.axhline(y=0, color='black', linewidth=1)
        ax4.axhline(y=1, color='green', linewidth=1,
                   linestyle='--', alpha=0.5, label='Sharpe = 1.0')
        ax4.set_title('Individual Stock Sharpe Ratios',
                     fontsize=13, fontweight='bold')
        ax4.set_xlabel('Stock', fontsize=11)
        ax4.set_ylabel('Sharpe Ratio', fontsize=11)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, sharpe in zip(bars, sharpes):
            ax4.text(bar.get_x() + bar.get_width()/2.,
                    bar.get_height() + 0.02,
                    f'{sharpe:.2f}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Portfolio Optimization Dashboard\n'
                    'Modern Portfolio Theory (Markowitz)',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('portfolio_optimization.png', dpi=300,
                   bbox_inches='tight')
        print("\nChart saved as 'portfolio_optimization.png'")
        plt.show()

    def run(self):
        """Run complete portfolio optimization"""
        self.download_data()
        self.calculate_returns()
        self.simulate_portfolios()
        self.find_optimal_portfolios()
        self.plot_results()

if __name__ == "__main__":
    optimizer = PortfolioOptimizer(
        tickers=[
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM',
            'JNJ', 'V', 'PG', 'XOM', 'BRK-B'
        ],
        num_portfolios=5000,
        risk_free_rate=0.05
    )
    optimizer.run()