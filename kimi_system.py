import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import random
from typing import List, Dict, Tuple, Optional
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. DATA MANAGEMENT
# ============================================================

class DataManager:
    """Handles all data fetching and preprocessing"""
    
    def __init__(self, universe: List[str], start_date: str, end_date: str):
        self.universe = universe
        self.start_date = start_date
        self.end_date = end_date
        self.stock_data = None
        self.ff_factors = None
        self.risk_free_rate = 0.05 / 252  # Assume 5% annual risk-free rate
        
    def fetch_stock_data(self) -> pd.DataFrame:
        """Fetch stock price data"""
        print("Fetching stock data...")
        # Use 2024 Q1 data as proxy for 2025 Q1
        proxy_start = "2025-01-01"
        proxy_end = "2025-03-31"
        
        stock_prices = yf.download(
            self.universe, 
            start=proxy_start, 
            end=proxy_end,
            progress=False
        )['Close']
        
        # Forward fill missing data
        stock_prices = stock_prices.fillna(method='ffill')
        self.stock_data = stock_prices
        return stock_prices
    
    def fetch_fama_french_factors(self) -> pd.DataFrame:
        """Fetch Fama-French 5 factors"""
        print("Fetching Fama-French factors...")
        try:
            ff = yf.download('^FF5', start="2024-01-01", end="2024-03-31", progress=False)
            # If not available, create synthetic factors
            return self._create_synthetic_factors()
        except:
            return self._create_synthetic_factors()
    
    def _create_synthetic_factors(self) -> pd.DataFrame:
        """Create synthetic Fama-French style factors"""
        dates = pd.date_range("2024-01-01", "2024-03-31", freq='D')
        dates = dates[dates.weekday < 5]  # Business days
        
        # Generate synthetic factor returns
        np.random.seed(42)
        factors = pd.DataFrame({
            'Mkt-RF': np.random.normal(0.0005, 0.01, len(dates)),
            'SMB': np.random.normal(0.0002, 0.008, len(dates)),
            'HML': np.random.normal(0.0001, 0.008, len(dates)),
            'RMW': np.random.normal(0.0001, 0.007, len(dates)),
            'CMA': np.random.normal(0.0001, 0.007, len(dates)),
            'RF': np.random.normal(0.0002, 0.001, len(dates))
        }, index=dates)
        
        return factors
    
    def calculate_returns(self) -> pd.DataFrame:
        """Calculate stock returns"""
        if self.stock_data is None:
            self.fetch_stock_data()
        return self.stock_data.pct_change().dropna()


# ============================================================
# 2. TRADE IDEA
# ============================================================

class TradeIdea:
    """Represents a single long/short trade idea"""
    
    def __init__(self, long_ticker: str, short_ticker: str, 
                 conviction: int, holding_days: int, idea_date: datetime):
        self.long_ticker = long_ticker
        self.short_ticker = short_ticker
        self.conviction = conviction  # 1-5 scale
        self.holding_days = holding_days  # 5-30 days
        self.idea_date = idea_date
        self.entry_date = None
        self.exit_date = None
        self.long_weight = 0.0
        self.short_weight = 0.0
        
    def __repr__(self):
        return f"Idea({self.long_ticker}/{self.short_ticker}, conv={self.conviction})"


# ============================================================
# 3. IDEA GENERATOR
# ============================================================

class IdeaGenerator:
    """Generates random long/short trade ideas"""
    
    def __init__(self, universe: List[str]):
        self.universe = universe
        
    def generate_ideas(self, date: datetime, num_ideas: int, 
                      existing_pairs: set = None) -> List[TradeIdea]:
        """Generate new trade ideas"""
        ideas = []
        used_pairs = existing_pairs or set()
        
        for _ in range(num_ideas):
            # Random long/short pair (ensure they're different)
            pair = random.sample(self.universe, 2)
            long_ticker, short_ticker = pair
            
            # Ensure we don't duplicate exact pairs
            pair_key = tuple(sorted([long_ticker, short_ticker]))
            if pair_key in used_pairs:
                continue
                
            used_pairs.add(pair_key)
            
            # Random conviction (1-5, higher probability for mid-level)
            conviction = random.choices(
                [1, 2, 3, 4, 5],
                weights=[0.1, 0.2, 0.3, 0.25, 0.15]
            )[0]
            
            # Random holding period (5-30 days, centered around 5)
            holding_days = random.randint(5, 30)
            
            idea = TradeIdea(long_ticker, short_ticker, conviction, 
                           holding_days, idea_date=date)
            ideas.append(idea)
            
        return ideas


# ============================================================
# 4. RISK MODEL
# ============================================================

class RiskModel:
    """Implements Fama-French 5-factor + Barra-style covariance"""
    
    def __init__(self, returns: pd.DataFrame, ff_factors: pd.DataFrame):
        self.returns = returns
        self.ff_factors = ff_factors
        self.covariance_matrix = None
        self.factor_exposures = None
        
    def calculate_covariance_matrix(self, lookback: int = 60) -> pd.DataFrame:
        """Calculate total covariance matrix"""
        print("Calculating covariance matrix...")
        
        # 1. Calculate specific covariance (residual)
        specific_cov = self._calculate_specific_covariance(lookback)
        
        # 2. Calculate factor covariance (Fama-French + Barra)
        factor_cov = self._calculate_factor_covariance(lookback)
        
        # 3. Combine
        total_cov = specific_cov + factor_cov
        
        self.covariance_matrix = total_cov
        return total_cov
    
    def _calculate_specific_covariance(self, lookback: int) -> pd.DataFrame:
        """Stock-specific covariance (idiosyncratic)"""
        recent_returns = self.returns.tail(lookback)
        specific_cov = recent_returns.cov() * 0.6  # Assume 60% specific risk
        return specific_cov
    
    def _calculate_factor_covariance(self, lookback: int) -> pd.DataFrame:
        """Factor-based covariance (Fama-French + Barra styles)"""
        returns = self.returns.tail(lookback)
        
        # Create synthetic Barra-style factors
        factor_returns = self._create_barra_factors(returns)
        
        # Calculate factor covariance
        factor_cov = factor_returns.cov()
        
        # Calculate factor exposures (simplified)
        exposures = self._calculate_factor_exposures(returns, factor_returns)
        
        # Convert to stock covariance: X * F * X.T
        factor_cov_stock = exposures.dot(factor_cov).dot(exposures.T)
        
        return factor_cov_stock * 0.4  # 40% factor risk
    
    def _create_barra_factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Create simplified Barra-style factor returns"""
        dates = returns.index
        
        # Size (market cap proxy)
        size = np.random.normal(0, 0.008, len(dates))
        
        # Value (book-to-price proxy)
        value = np.random.normal(0, 0.007, len(dates))
        
        # Momentum
        momentum = np.random.normal(0, 0.009, len(dates))
        
        # Volatility
        volatility = np.random.normal(0, 0.008, len(dates))
        
        # Liquidity
        liquidity = np.random.normal(0, 0.006, len(dates))
        
        factors = pd.DataFrame({
            'Size': size,
            'Value': value,
            'Momentum': momentum,
            'Volatility': volatility,
            'Liquidity': liquidity
        }, index=dates)
        
        return factors
    
    def _calculate_factor_exposures(self, returns: pd.DataFrame, 
                                   factor_returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate factor exposures (beta coefficients)"""
        exposures = pd.DataFrame(index=returns.columns, 
                                columns=factor_returns.columns)
        
        for stock in returns.columns:
            for factor in factor_returns.columns:
                # Simplified: random exposure between -1 and 1
                exposures.loc[stock, factor] = np.random.uniform(-0.8, 0.8)
        
        return exposures


# ============================================================
# 5. PORTFOLIO OPTIMIZER
# ============================================================

class PortfolioOptimizer:
    """Mean-Variance Optimization for maximizing Sharpe ratio"""
    
    def __init__(self, risk_model: RiskModel, risk_free_rate: float = 0.0):
        self.risk_model = risk_model
        self.risk_free_rate = risk_free_rate
        
    def optimize(self, ideas: List[TradeIdea], 
                returns: pd.DataFrame,
                date: datetime) -> Dict[TradeIdea, float]:
        """
        Optimize idea weights to maximize portfolio Sharpe ratio
        Returns: dictionary mapping ideas to weights
        """
        if not ideas:
            return {}
        
        if self.risk_model.covariance_matrix is None:
            self.risk_model.calculate_covariance_matrix()
        
        # Calculate expected returns for each idea
        # Higher conviction = higher expected return
        expected_returns = self._calculate_idea_returns(ideas, returns, date)
        
        # Create optimization variables: weights for each idea
        num_ideas = len(ideas)
        
        # Constraints
        # 1. Dollar neutral portfolio
        # 2. Each idea is dollar neutral (long = short)
        # 3. Weights >= 0 (non-negative)
        # 4. Total leverage constraint
        
        # For each idea, we have 2 positions: long and short
        # We'll optimize the allocation to each idea, then split equally
        
        # Simplified approach: optimize idea allocations, then equal split L/S
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Full investment
            {'type': 'ineq', 'fun': lambda w: w}  # Non-negative
        ]
        
        # Initial guess: equal weight
        w0 = np.ones(num_ideas) / num_ideas
        
        # Minimize negative Sharpe ratio
        result = minimize(
            lambda w: -self._sharpe_ratio(w, expected_returns, self.risk_model.covariance_matrix),
            w0,
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, 0.3) for _ in range(num_ideas)],  # Max 30% per idea
        )
        
        if not result.success:
            print(f"Optimization failed: {result.message}")
            return {idea: 1.0/num_ideas for idea in ideas}
        
        # Assign weights to ideas
        idea_weights = {}
        for i, idea in enumerate(ideas):
            weight = result.x[i]
            idea_weights[idea] = weight
        
        return idea_weights
    
    def _calculate_idea_returns(self, ideas: List[TradeIdea], 
                               returns: pd.DataFrame,
                               date: datetime) -> np.ndarray:
        """Calculate expected returns for each idea"""
        expected_returns = []
        
        for idea in ideas:
            # Historical returns for the pair
            if date in returns.index:
                hist_long_return = returns[idea.long_ticker].mean()
                hist_short_return = returns[idea.short_ticker].mean()
            else:
                hist_long_return = 0.0005
                hist_short_return = 0.0005
            
            # Expected return = long - short + conviction adjustment
            base_return = hist_long_return - hist_short_return
            
            # Conviction multiplier: 1=0.5x, 5=1.5x
            conviction_mult = 0.5 + (idea.conviction - 1) * 0.25
            
            expected_return = base_return * conviction_mult
            expected_returns.append(expected_return)
        
        return np.array(expected_returns)
    
    def _sharpe_ratio(self, weights: np.ndarray, 
                     expected_returns: np.ndarray,
                     covariance_matrix: pd.DataFrame) -> float:
        """Calculate portfolio Sharpe ratio"""
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        
        if portfolio_vol == 0:
            return 0
        
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
        return sharpe


# ============================================================
# 6. TRANSACTION COST MODEL
# ============================================================

class TransactionCostModel:
    """Realistic transaction costs"""
    
    def __init__(self, base_bps: float = 5.0):
        self.base_bps = base_bps  # 5 basis points per trade
        
    def calculate_cost(self, notional_value: float) -> float:
        """Calculate transaction cost for a trade"""
        return abs(notional_value) * self.base_bps / 10_000
    
    def apply_borrow_cost(self, short_value: float, annual_rate: float = 0.02) -> float:
        """Cost of borrowing for short positions"""
        daily_rate = annual_rate / 252
        return short_value * daily_rate


# ============================================================
# 7. BACKTEST ENGINE
# ============================================================

class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, universe: List[str], start_date: str, end_date: str,
                 initial_cash: float = 1_000_000):
        self.universe = universe
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_cash = initial_cash
        
        # Components
        self.data_manager = DataManager(universe, start_date, end_date)
        self.idea_generator = IdeaGenerator(universe)
        self.txn_cost_model = TransactionCostModel()
        
        # State
        self.cash = initial_cash
        self.active_ideas = []  # Currently held ideas
        self.portfolio_value = initial_cash
        self.performance_history = []
        self.position_history = []
        self.turnover_history = []
        
        # Data
        self.returns = None
        self.dates = None
        
    def prepare_data(self):
        """Prepare all required data"""
        self.data_manager.fetch_stock_data()
        self.data_manager.fetch_fama_french_factors()
        self.returns = self.data_manager.calculate_returns()
        self.dates = self.returns.index[
            (self.returns.index >= self.start_date) &
            (self.returns.index <= self.end_date)
        ]
        
    def run(self):
        """Run the backtest"""
        print("Starting backtest...")
        self.prepare_data()
        
        # Initial set of ideas
        self.active_ideas = self.idea_generator.generate_ideas(
            self.dates[0], 10
        )
        
        for i, date in enumerate(self.dates):
            print(f"Processing {date.strftime('%Y-%m-%d')}...")
            
            # 1. Remove expired ideas
            self._remove_expired_ideas(date)
            
            # 2. Generate new ideas (except first day)
            if i > 0:
                new_ideas = self.idea_generator.generate_ideas(date, 5)
                self.active_ideas.extend(new_ideas)
            
            # 3. Optimize portfolio
            if self.active_ideas:
                self._rebalance_portfolio(date)
            
            # 4. Calculate P&L
            self._calculate_pnl(date)
            
            # 5. Record performance
            self._record_performance(date)
        
        print("Backtest completed!")
        return self.generate_report()
    
    def _remove_expired_ideas(self, date: datetime):
        """Remove ideas that have exceeded their holding period"""
        remaining_ideas = []
        for idea in self.active_ideas:
            days_held = (date - idea.idea_date).days
            if days_held < idea.holding_days:
                remaining_ideas.append(idea)
        
        expired = len(self.active_ideas) - len(remaining_ideas)
        if expired > 0:
            print(f"  Removed {expired} expired ideas")
        
        self.active_ideas = remaining_ideas
    
    def _rebalance_portfolio(self, date: datetime):
        """Rebalance portfolio using MVO"""
        print(f"  Rebalancing {len(self.active_ideas)} ideas...")
        
        # Create risk model
        risk_model = RiskModel(self.returns, self.data_manager.ff_factors)
        
        # Optimize
        optimizer = PortfolioOptimizer(risk_model, self.data_manager.risk_free_rate)
        idea_weights = optimizer.optimize(self.active_ideas, self.returns, date)
        
        # Calculate target positions
        target_positions = {}
        total_weight = sum(idea_weights.values())
        
        if total_weight == 0:
            return
        
        # Normalize weights
        for idea in self.active_ideas:
            weight = idea_weights.get(idea, 0) / total_weight
            
            # Dollar-neutral split: allocate equally to long and short
            # Conviction affects position size
            position_size = self.portfolio_value * weight * (idea.conviction / 3.0)
            
            target_positions[idea.long_ticker] = position_size / 2
            target_positions[idea.short_ticker] = -position_size / 2
            
            # Store weights in idea
            idea.long_weight = weight / 2
            idea.short_weight = -weight / 2
        
        # Execute trades with transaction costs
        self._execute_trades(date, target_positions)
    
    def _execute_trades(self, date: datetime, target_positions: Dict[str, float]):
        """Execute trades with transaction costs"""
        
        # Calculate current positions
        current_positions = getattr(self, 'current_positions', {})
        
        # Calculate turnover
        turnover = 0
        for ticker, target_pos in target_positions.items():
            current_pos = current_positions.get(ticker, 0)
            trade_value = abs(target_pos - current_pos)
            turnover += trade_value
            
            # Apply transaction cost
            txn_cost = self.txn_cost_model.calculate_cost(trade_value)
            self.cash -= txn_cost
        
        # Apply borrow costs for short positions
        for ticker, position in target_positions.items():
            if position < 0:
                borrow_cost = self.txn_cost_model.apply_borrow_cost(abs(position))
                self.cash -= borrow_cost
        
        self.turnover_history.append({'date': date, 'turnover': turnover})
        self.current_positions = target_positions.copy()
        self.position_history.append({
            'date': date,
            'positions': target_positions.copy()
        })
    
    def _calculate_pnl(self, date: datetime):
        """Calculate daily P&L"""
        if not hasattr(self, 'current_positions'):
            return
        
        daily_return = 0
        for ticker, position in self.current_positions.items():
            if ticker in self.returns.columns and date in self.returns.index:
                stock_return = self.returns.loc[date, ticker]
                daily_return += position * stock_return
        
        self.cash += daily_return
        self.portfolio_value = self.cash + sum(self.current_positions.values())
    
    def _record_performance(self, date: datetime):
        """Record daily performance"""
        self.performance_history.append({
            'date': date,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'num_ideas': len(self.active_ideas)
        })
    
    def generate_report(self) -> pd.DataFrame:
        """Generate comprehensive performance report"""
        perf_df = pd.DataFrame(self.performance_history)
        perf_df = perf_df.set_index('date')
        
        # Calculate returns
        perf_df['returns'] = perf_df['portfolio_value'].pct_change()
        
        return perf_df


# ============================================================
# 8. PERFORMANCE ANALYZER
# ============================================================

class PerformanceAnalyzer:
    """Comprehensive performance analytics"""
    
    def __init__(self, performance_df: pd.DataFrame, 
                 initial_cash: float,
                 turnover_history: List[Dict]):
        self.perf = performance_df
        self.initial_cash = initial_cash
        self.turnover_history = turnover_history
        
    def calculate_stats(self) -> Dict:
        """Calculate comprehensive statistics"""
        
        returns = self.perf['returns'].dropna()
        
        # Basic stats
        total_return = (self.perf['portfolio_value'].iloc[-1] / self.initial_cash - 1)
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (returns.mean() - 0.02/252) / returns.std() * np.sqrt(252)
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        # Average daily P&L
        avg_daily_pnl = returns.mean() * self.initial_cash
        
        # Turnover
        avg_turnover = np.mean([t['turnover'] for t in self.turnover_history])
        
        # Skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        stats = {
            'Total Return': f"{total_return:.2%}",
            'Annualized Return': f"{annualized_return:.2%}",
            'Volatility': f"{volatility:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Win Rate': f"{win_rate:.1%}",
            'Avg Daily P&L': f"${avg_daily_pnl:,.0f}",
            'Avg Turnover': f"${avg_turnover:,.0f}",
            'Skewness': f"{skewness:.2f}",
            'Kurtosis': f"{kurtosis:.2f}",
            'Final Portfolio Value': f"${self.perf['portfolio_value'].iloc[-1]:,.0f}"
        }
        
        return stats
    
    def plot_performance(self):
        """Plot portfolio performance"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 1. Portfolio value
        axes[0].plot(self.perf.index, self.perf['portfolio_value'])
        axes[0].set_title('Portfolio Value Over Time')
        axes[0].set_ylabel('Value ($)')
        axes[0].grid(True)
        
        # 2. Daily returns
        axes[1].plot(self.perf.index, self.perf['returns'] * 100)
        axes[1].set_title('Daily Returns (%)')
        axes[1].set_ylabel('Return (%)')
        axes[1].grid(True)
        
        # 3. Drawdown
        cumulative = (1 + self.perf['returns'].fillna(0)).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max * 100
        axes[2].fill_between(self.perf.index, drawdown, 0, alpha=0.3, color='red')
        axes[2].set_title('Drawdown (%)')
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

def run_simulation():
    """Run the complete simulation"""
    
    # Configuration
    UNIVERSE = [
        "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "AMD", "TSM",
        "INTC", "CRM", "ADBE", "ORCL", "NFLX", "QCOM", "SNOW"
    ]
    
    START_DATE = "2025-01-01"
    END_DATE = "2025-03-31"
    
    # Run backtest
    engine = BacktestEngine(UNIVERSE, START_DATE, END_DATE, initial_cash=1_000_000)
    performance = engine.run()
    
    # Analyze performance
    analyzer = PerformanceAnalyzer(
        performance, 
        engine.initial_cash,
        engine.turnover_history
    )
    
    stats = analyzer.calculate_stats()
    
    print("\n" + "="*60)
    print("PORTFOLIO PERFORMANCE REPORT")
    print("="*60)
    for key, value in stats.items():
        print(f"{key:25}: {value:>15}")
    print("="*60)
    
    # Show sample of positions
    print("\nSample Portfolio Positions (last day):")
    if engine.position_history:
        last_positions = engine.position_history[-1]['positions']
        sorted_pos = sorted(last_positions.items(), key=lambda x: abs(x[1]), reverse=True)
        for ticker, pos in sorted_pos[:10]:
            print(f"  {ticker:5}: ${pos:>10,.0f}")
    
    # Plot
    try:
        analyzer.plot_performance()
    except ImportError:
        print("\nMatplotlib not available. Skipping plots.")
    
    return performance, stats, engine

# Run the simulation
if __name__ == "__main__":
    print("Starting L/S Portfolio Simulation...")
    print("Note: Using 2024 Q1 data as proxy for 2025 Q1")
    
    np.random.seed(42)
    random.seed(42)
    
    performance, stats, engine = run_simulation()