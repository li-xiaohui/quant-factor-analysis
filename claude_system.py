import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings
from pandas_datareader.data import DataReader
warnings.filterwarnings('ignore')

# Configuration
STOCKS = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "AMD", "TSM", 
          "INTC", "CRM", "ADBE", "ORCL", "NFLX", "QCOM", "SNOW"]
START_DATE = "2024-10-01"  # Extended for factor estimation
SIM_START = "2025-01-01"
END_DATE = "2025-03-31"
INITIAL_CAPITAL = 10_000_000  # $10M
TRANSACTION_COST_BPS = 10  # 10 bps
MAX_HOLDING_DAYS = 21  # 1 month
AVG_HOLDING_DAYS = 5
INITIAL_IDEAS = 10
DAILY_NEW_IDEAS = 5

print("=" * 80)
print("TMT LONG/SHORT PORTFOLIO SIMULATION")
print("=" * 80)
print(f"Universe: {len(STOCKS)} stocks")
print(f"Period: {SIM_START} to {END_DATE}")
print(f"Initial Capital: ${INITIAL_CAPITAL:,.0f}")
print(f"Transaction Cost: {TRANSACTION_COST_BPS} bps")
print("=" * 80)

# ============================================================================
# 1. DATA ACQUISITION
# ============================================================================
print("\n[1/6] Downloading price data...")

data = yf.download(STOCKS, start=START_DATE, end=END_DATE, progress=False)['Close']
data = data.ffill().bfill()  # Handle missing data

# Calculate returns
returns = data.pct_change().dropna()
print(f"✓ Downloaded {len(data)} days of data for {len(STOCKS)} stocks")

# Download Fama-French 5 factors (simulated with market proxies)
print("\n[2/6] Constructing Fama-French 5 Factor model...")

# For simulation, we'll use proxies:
# SPY for market, and construct other factors from our universe
spy_data = yf.download('SPY', start=START_DATE, end=END_DATE, progress=False)['Close']
spy_returns = spy_data.pct_change().dropna()

# Align dates
common_dates = returns.index.intersection(spy_returns.index)
returns = returns.loc[common_dates]
spy_returns = spy_returns.loc[common_dates]

# Construct synthetic FF5 factors
rf = 0.05 / 252  # Risk-free rate (5% annual)
mkt_rf = spy_returns - rf
# Use pandas-datareader to download Fama-French 5 factors instead of simulating

try:
    # Download monthly FF5 (percent values) and convert to daily by forward-filling
    ff_raw = DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench',
                        start=START_DATE, end=END_DATE)
    ff_month = ff_raw[0].copy()

    # Ensure timestamp index (period -> timestamp)
    if hasattr(ff_month.index, "to_timestamp"):
        ff_month.index = ff_month.index.to_timestamp()
    else:
        ff_month.index = pd.to_datetime(ff_month.index)

    # Keep relevant columns (FF dataset uses percent values)
    cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
    ff_month = ff_month[cols]

    # Reindex to our trading calendar (daily) and forward-fill monthly values
    ff_daily = ff_month.reindex(common_dates).ffill()

    # Convert percent -> decimal and approximate daily from monthly by dividing by ~21 trading days
    ff_daily = ff_daily / 100.0 / 21.0

    # Build factor DataFrame (use Mkt-RF from FF directly)
    mkt_rf = ff_daily['Mkt-RF']  # overrides prior mkt_rf variable to align with FF data
    ff5_factors = ff_daily[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].loc[common_dates].copy()

except Exception as e:
    # Fallback to simulated factors if download fails
    print(f"Warning: failed to download Fama-French data ({e}), using simulated factors.")
    market_caps = {s: np.random.uniform(100, 3000) for s in STOCKS}  # Billions
    smb = returns[sorted(STOCKS, key=lambda x: market_caps[x])[:5]].mean(axis=1) - \
          returns[sorted(STOCKS, key=lambda x: market_caps[x])[-5:]].mean(axis=1)

    hml = returns[STOCKS[:5]].mean(axis=1) - returns[STOCKS[-5:]].mean(axis=1)

    rmw = returns[['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA']].mean(axis=1) - \
          returns[['INTC', 'SNOW']].mean(axis=1)

    cma = returns[['MSFT', 'ORCL', 'INTC']].mean(axis=1) - \
          returns[['NVDA', 'AMD', 'SNOW']].mean(axis=1)

    ff5_factors = pd.DataFrame({
        'Mkt-RF': mkt_rf,
        'SMB': smb,
        'HML': hml,
        'RMW': rmw,
        'CMA': cma
    }, index=common_dates)

print(f"✓ Constructed FF5 factors: {list(ff5_factors.columns)}")

# Estimate factor loadings (betas) for each stock
factor_loadings = {}
for stock in STOCKS:
    y = returns[stock] - rf
    X = ff5_factors
    X = X.loc[y.index]
    
    # OLS regression
    X_with_const = np.column_stack([np.ones(len(X)), X.values])
    betas = np.linalg.lstsq(X_with_const, y.values, rcond=None)[0]
    
    factor_loadings[stock] = {
        'alpha': betas[0],
        'Mkt-RF': betas[1],
        'SMB': betas[2],
        'HML': betas[3],
        'RMW': betas[4],
        'CMA': betas[5]
    }

print(f"✓ Estimated factor loadings for {len(STOCKS)} stocks")

# ============================================================================
# 3. BARRA-STYLE COVARIANCE MATRIX
# ============================================================================
print("\n[3/6] Building Barra-style covariance matrix...")

# Factor covariance matrix
factor_cov = ff5_factors.cov() * 252  # Annualized

# Specific risk (idiosyncratic)
specific_var = {}
for stock in STOCKS:
    # Residual variance from factor model
    y = returns[stock] - rf
    X = ff5_factors.loc[y.index]
    X_with_const = np.column_stack([np.ones(len(X)), X.values])
    y_pred = X_with_const @ np.array([factor_loadings[stock]['alpha']] + 
                                      [factor_loadings[stock][f] for f in ff5_factors.columns])
    residuals = y - y_pred
    specific_var[stock] = residuals.var() * 252  # Annualized

print(f"✓ Built factor covariance matrix ({ff5_factors.shape[1]}x{ff5_factors.shape[1]})")
print(f"✓ Calculated specific risks for {len(STOCKS)} stocks")

# ============================================================================
# 4. TRADE IDEA GENERATION
# ============================================================================
print("\n[4/6] Generating trade ideas...")

class TradeIdea:
    def __init__(self, idea_id, idea_date, long_stock, short_stock, conviction, 
                 holding_days, expected_return):
        self.idea_id = idea_id
        self.idea_date = idea_date
        self.long_stock = long_stock
        self.short_stock = short_stock
        self.conviction = conviction
        self.holding_days = holding_days
        self.expected_return = expected_return
        self.entry_date = None
        self.exit_date = None
        self.pnl = 0.0
    
    def is_expired(self, current_date):
        if self.entry_date is None:
            return False
        days_held = (current_date - self.entry_date).days
        return days_held >= self.holding_days

def generate_ideas(num_ideas, idea_date, idea_counter):
    """Generate random but plausible trade ideas"""
    ideas = []
    
    for _ in range(num_ideas):
        # Random long/short pair
        long_stock = np.random.choice(STOCKS)
        short_stock = np.random.choice([s for s in STOCKS if s != long_stock])
        
        # Conviction level (1-5)
        conviction = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.4, 0.2, 0.1])
        
        # Holding period (skewed towards AVG_HOLDING_DAYS)
        holding_days = min(MAX_HOLDING_DAYS, 
                          max(1, int(np.random.gamma(2, AVG_HOLDING_DAYS/2))))
        
        # Expected return (higher conviction = higher expected return)
        expected_return = conviction * 0.002 * np.random.uniform(0.5, 1.5)  # 0.2-1.5% per conviction level
        
        idea = TradeIdea(
            idea_id=idea_counter,
            idea_date=idea_date,
            long_stock=long_stock,
            short_stock=short_stock,
            conviction=conviction,
            holding_days=holding_days,
            expected_return=expected_return
        )
        ideas.append(idea)
        idea_counter += 1
    
    return ideas, idea_counter

# Generate initial ideas
sim_dates = returns.loc[SIM_START:END_DATE].index
trading_dates = sim_dates.tolist()

all_ideas = []
idea_counter = 1

# Initial ideas (day before sim start)
initial_ideas, idea_counter = generate_ideas(INITIAL_IDEAS, trading_dates[0] - timedelta(days=1), idea_counter)
all_ideas.extend(initial_ideas)

print(f"✓ Generated {INITIAL_IDEAS} initial ideas")

# ============================================================================
# 5. PORTFOLIO OPTIMIZATION & SIMULATION
# ============================================================================
print("\n[5/6] Running portfolio simulation with MVO...")

def calculate_idea_return(idea, prices_long, prices_short):
    """Calculate dollar-neutral L/S return"""
    ret_long = prices_long.pct_change().fillna(0)
    ret_short = prices_short.pct_change().fillna(0)
    return 0.5 * ret_long - 0.5 * ret_short  # Dollar neutral

def build_idea_covariance_matrix(active_ideas, returns_hist, factor_loadings, factor_cov, specific_var):
    """Build covariance matrix for active ideas using Barra-style approach"""
    n = len(active_ideas)
    cov_matrix = np.zeros((n, n))
    
    for i, idea_i in enumerate(active_ideas):
        for j, idea_j in enumerate(active_ideas):
            # Factor contribution
            factor_cov_ij = 0
            for factor in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']:
                beta_i = (0.5 * factor_loadings[idea_i.long_stock][factor] - 
                         0.5 * factor_loadings[idea_i.short_stock][factor])
                beta_j = (0.5 * factor_loadings[idea_j.long_stock][factor] - 
                         0.5 * factor_loadings[idea_j.short_stock][factor])
                factor_cov_ij += beta_i * beta_j * factor_cov.loc[factor, factor]
            
            # Specific risk contribution (only diagonal)
            specific_cov_ij = 0
            if i == j:
                specific_cov_ij = (0.25 * specific_var[idea_i.long_stock] + 
                                  0.25 * specific_var[idea_i.short_stock])
            
            cov_matrix[i, j] = factor_cov_ij + specific_cov_ij
    
    return cov_matrix

def optimize_portfolio(active_ideas, returns_hist, factor_loadings, factor_cov, specific_var):
    """Mean-Variance Optimization to maximize Sharpe ratio"""
    if len(active_ideas) == 0:
        return np.array([])
    
    n = len(active_ideas)
    
    # Expected returns (based on conviction and historical performance)
    expected_returns = np.array([idea.expected_return * idea.conviction for idea in active_ideas])
    
    # Covariance matrix
    cov_matrix = build_idea_covariance_matrix(active_ideas, returns_hist, 
                                              factor_loadings, factor_cov, specific_var)
    
    # Add small regularization to ensure positive definite
    cov_matrix += np.eye(n) * 1e-6
    
    # Optimization: maximize Sharpe ratio = max (w'μ - rf) / sqrt(w'Σw)
    # Equivalent to: min -w'μ / sqrt(w'Σw) subject to sum(w) = 1, w >= 0
    
    def neg_sharpe(w):
        port_return = np.dot(w, expected_returns)
        port_vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
        return -port_return / (port_vol + 1e-8)
    
    # Constraints: weights sum to 1, all non-negative
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n)]
    
    # Initial guess: equal weight
    w0 = np.ones(n) / n
    
    # Optimize
    result = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, 
                     constraints=constraints, options={'maxiter': 1000})
    
    if result.success:
        return result.x
    else:
        # Fallback to equal weight
        return np.ones(n) / n

# Simulation state
portfolio_value = INITIAL_CAPITAL
cash = INITIAL_CAPITAL
active_ideas = []
daily_pnl = []
daily_values = []
daily_positions = []
transaction_costs = []

# Track performance
performance_log = []

for i, current_date in enumerate(trading_dates):
    # Generate new ideas daily (except first day)
    if i > 0:
        new_ideas, idea_counter = generate_ideas(DAILY_NEW_IDEAS, current_date, idea_counter)
        all_ideas.extend(new_ideas)
    
    # Add today's new ideas to active pool (they become tradeable next day)
    for idea in all_ideas:
        if idea.idea_date < current_date and idea.entry_date is None:
            if idea not in active_ideas:
                active_ideas.append(idea)
                idea.entry_date = current_date
    
    # Remove expired ideas
    expired_ideas = [idea for idea in active_ideas if idea.is_expired(current_date)]
    for idea in expired_ideas:
        idea.exit_date = current_date
        active_ideas.remove(idea)
    
    # Portfolio optimization
    if len(active_ideas) > 0:
        # Get historical returns for covariance estimation (last 60 days)
        lookback = min(60, i + len(returns.loc[:SIM_START]))
        returns_hist = returns.iloc[-lookback:]
        
        # Optimize
        weights = optimize_portfolio(active_ideas, returns_hist, factor_loadings, 
                                     factor_cov, specific_var)
        
        # Calculate positions for each idea (dollar neutral)
        daily_position = {}
        tc_today = 0
        
        for idea, weight in zip(active_ideas, weights):
            allocation = portfolio_value * weight
            
            # Long leg
            long_price = data.loc[current_date, idea.long_stock]
            long_shares = allocation / (2 * long_price)
            
            # Short leg
            short_price = data.loc[current_date, idea.short_stock]
            short_shares = -allocation / (2 * short_price)
            
            daily_position[f"{idea.idea_id}_long_{idea.long_stock}"] = long_shares
            daily_position[f"{idea.idea_id}_short_{idea.short_stock}"] = short_shares
            
            # Transaction costs (only on rebalancing)
            if i > 0:
                tc_today += allocation * TRANSACTION_COST_BPS / 10000
        
        transaction_costs.append(tc_today)
        cash -= tc_today
    else:
        daily_position = {}
        transaction_costs.append(0)
    
    daily_positions.append(daily_position)
    
    # Calculate P&L
    if i > 0:
        pnl = 0
        prev_position = daily_positions[i-1]
        
        for key, shares in prev_position.items():
            parts = key.split('_')
            ticker = parts[-1]
            
            if current_date in data.index:
                price_prev = data.loc[trading_dates[i-1], ticker]
                price_curr = data.loc[current_date, ticker]
                pnl += shares * (price_curr - price_prev)
        
        daily_pnl.append(pnl)
        portfolio_value = portfolio_value + pnl - transaction_costs[-1]
    else:
        daily_pnl.append(0)
    
    daily_values.append(portfolio_value)
    
    # Log
    performance_log.append({
        'date': current_date,
        'portfolio_value': portfolio_value,
        'num_active_ideas': len(active_ideas),
        'num_expired': len(expired_ideas),
        'cash': cash,
        'pnl': daily_pnl[-1],
        'transaction_cost': transaction_costs[-1]
    })

print(f"✓ Simulated {len(trading_dates)} trading days")
print(f"✓ Generated {len(all_ideas)} total ideas")
print(f"✓ Total transaction costs: ${sum(transaction_costs):,.2f}")

# ============================================================================
# 6. PERFORMANCE ANALYSIS
# ============================================================================
print("\n[6/6] Calculating performance statistics...\n")

performance_df = pd.DataFrame(performance_log)
performance_df.set_index('date', inplace=True)

# Calculate returns
portfolio_returns = performance_df['portfolio_value'].pct_change().dropna()

# Performance metrics
total_return = (daily_values[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
annualized_return = (1 + total_return) ** (252 / len(trading_dates)) - 1
volatility = portfolio_returns.std() * np.sqrt(252)
sharpe_ratio = (annualized_return - 0.05) / volatility if volatility > 0 else 0

# Drawdown
cumulative = (1 + portfolio_returns).cumprod()
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min()

# Win rate
winning_days = (portfolio_returns > 0).sum()
win_rate = winning_days / len(portfolio_returns)

# Information Ratio (vs SPY)
spy_sim = yf.download('SPY', start=SIM_START, end=END_DATE, progress=False)['Close']
spy_returns_sim = spy_sim.pct_change().dropna()
common_dates_sim = portfolio_returns.index.intersection(spy_returns_sim.index)
active_return = portfolio_returns.loc[common_dates_sim] - spy_returns_sim.loc[common_dates_sim]
information_ratio = active_return.mean() / active_return.std() * np.sqrt(252) if active_return.std() > 0 else 0

print("=" * 80)
print("PORTFOLIO PERFORMANCE SUMMARY")
print("=" * 80)
print(f"Initial Capital:        ${INITIAL_CAPITAL:,.0f}")
print(f"Final Portfolio Value:  ${daily_values[-1]:,.0f}")
print(f"Total Return:           {total_return*100:.2f}%")
print(f"Annualized Return:      {annualized_return*100:.2f}%")
print(f"Annualized Volatility:  {volatility*100:.2f}%")
print(f"Sharpe Ratio:           {sharpe_ratio:.3f}")
print(f"Information Ratio:      {information_ratio:.3f}")
print(f"Maximum Drawdown:       {max_drawdown*100:.2f}%")
print(f"Win Rate:               {win_rate*100:.2f}%")
print(f"Total Trading Days:     {len(trading_dates)}")
print(f"Total Transaction Cost: ${sum(transaction_costs):,.2f}")
print(f"Avg Daily TC:           ${np.mean(transaction_costs):,.2f}")
print("=" * 80)

print("\nIDEA GENERATION STATISTICS")
print("=" * 80)
print(f"Total Ideas Generated:  {len(all_ideas)}")
print(f"Avg Holding Period:     {np.mean([idea.holding_days for idea in all_ideas]):.1f} days")
print(f"Avg Conviction:         {np.mean([idea.conviction for idea in all_ideas]):.2f}")

conviction_dist = pd.Series([idea.conviction for idea in all_ideas]).value_counts().sort_index()
print(f"\nConviction Distribution:")
for conv, count in conviction_dist.items():
    print(f"  Level {conv}: {count} ideas ({count/len(all_ideas)*100:.1f}%)")

print("=" * 80)

print("\nTOP 10 PERFORMING DAYS")
print("=" * 80)
top_days = performance_df.nlargest(10, 'pnl')[['pnl', 'num_active_ideas', 'portfolio_value']]
top_days['pnl'] = top_days['pnl'].apply(lambda x: f"${x:,.0f}")
top_days['portfolio_value'] = top_days['portfolio_value'].apply(lambda x: f"${x:,.0f}")
print(top_days.to_string())

print("\n" + "=" * 80)
print("WORST 10 PERFORMING DAYS")
print("=" * 80)
worst_days = performance_df.nsmallest(10, 'pnl')[['pnl', 'num_active_ideas', 'portfolio_value']]
worst_days['pnl'] = worst_days['pnl'].apply(lambda x: f"${x:,.0f}")
worst_days['portfolio_value'] = worst_days['portfolio_value'].apply(lambda x: f"${x:,.0f}")
print(worst_days.to_string())

print("\n" + "=" * 80)
print("MONTHLY PERFORMANCE")
print("=" * 80)
monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
monthly_stats = pd.DataFrame({
    'Return': monthly_returns.values,
    'Volatility': portfolio_returns.resample('M').std().values * np.sqrt(21)
})
monthly_stats.index = monthly_returns.index.strftime('%Y-%m')
monthly_stats['Return'] = monthly_stats['Return'].apply(lambda x: f"{x*100:.2f}%")
monthly_stats['Volatility'] = monthly_stats['Volatility'].apply(lambda x: f"{x*100:.2f}%")
print(monthly_stats.to_string())

print("\n" + "=" * 80)
print("SIMULATION COMPLETE")
print("=" * 80)