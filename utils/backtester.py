"""
Offline backtesting framework for candlestick trading strategies.

This module provides a robust framework for backtesting trading strategies
against historical market data, with comprehensive performance metrics and
risk analysis capabilities.
"""
from utils.logger import setup_logger
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any, Union # Added Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# Import the centralized data validator
from core.data_validator import get_global_validator, validate_dataframe
# Import core indicator functions
from core.indicators.rsi import calculate_rsi
from core.indicators.macd import calculate_macd

# Configure logging
logger = setup_logger(__name__)

def empty_backtest_result(initial_capital: float) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Return zeroed‐out metrics, an empty equity curve, and empty trade log.
    """
    # Zero metrics
    metrics = {
        "total_return":     0.0,
        "sharpe_ratio":     0.0,
        "max_drawdown":     0.0,
        "win_rate":         0.0,
        "profit_factor":    0.0,
        "num_trades":       0,
        "avg_trade_return": 0.0
    }
    # Empty equity curve
    equity_curve = pd.DataFrame(columns=["date","equity"])
    # Empty trade log
    trade_log    = pd.DataFrame(columns=["date","symbol","side","price","quantity","fees"])
    return metrics, equity_curve, trade_log

@dataclass
class Trade:
    """Represents a single trade execution."""
    date: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    price: float
    quantity: int
    fees: float = 0.0

    def __post_init__(self):
        if self.side not in ['BUY', 'SELL']:
            raise ValueError("Trade side must be either 'BUY' or 'SELL'")
        if self.price <= 0:
            raise ValueError("Trade price must be positive")
        if self.quantity <= 0:
            raise ValueError("Trade quantity must be positive")

class BacktestConfig(BaseModel):
    """Configuration parameters for backtest execution."""
    initial_capital: float = Field(gt=0)
    commission_rate: float = Field(ge=0, le=0.01, default=0.001)
    slippage_rate: float = Field(ge=0, le=0.01, default=0.0005)
    risk_free_rate: float = Field(ge=0, le=0.1, default=0.02)
    position_size_limit: float = Field(ge=0, le=1.0, default=0.2)

    class Config:
        arbitrary_types_allowed = True

class BacktestResults(BaseModel):
    """Container for backtest performance metrics."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float

class Backtest:
    """Backtesting engine for trading strategies."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.reset()
        
    def reset(self):
        self.equity_curve: pd.Series = pd.Series()
        self.trades: List[Trade] = []
        self.positions: Dict[str, int] = {}
        self.capital: float = self.config.initial_capital
        
    def validate_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Validate market data using the centralized data validator.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
        
        Raises:
            ValueError: If any DataFrame fails validation
        """
        validator = get_global_validator()
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for symbol, df in data.items():
            # Validate DataFrame structure and integrity
            result = validator.validate_dataframe(
                df, 
                required_cols=required_columns,
                check_ohlcv=True, # Renamed from validate_ohlc
                detect_anomalies_level="basic" # Renamed from check_statistical_anomalies and passed a string value
            )
            
            # Check for DatetimeIndex specifically
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"DataFrame for {symbol} must have DatetimeIndex")
            
            # Handle validation errors
            if not result.is_valid:
                error_messages = result.errors if result.errors is not None else ["Unknown validation error"]
                raise ValueError(f"Invalid data for {symbol}: {', '.join(error_messages)}")
            
            # Display warnings but don't halt execution
            if result.warnings:
                for warning in result.warnings:
                    logger.warning(f"{symbol}: {warning}")

    def simulate(
        self,
        data: Dict[str, pd.DataFrame],
        strategy_fn: Callable[[pd.DataFrame], float] # Reverted to float, as simulate processes one signal at a time
    ) -> BacktestResults:
        try:
            self.validate_data(data)
            self.reset()
            self.positions = {sym: 0 for sym in data}

            dates = pd.date_range(
                start=min(df.index.min() for df in data.values()),
                end=max(df.index.max() for df in data.values()),
                freq='D'
            )
            equity_history = []

            for date in dates:
                daily_value = self.capital
                for symbol, df_symbol_data in data.items(): # Renamed df to df_symbol_data to avoid conflict
                    if date in df_symbol_data.index:
                        try:
                            # Get the market data for the current date. Ensure it's a Series.
                            current_market_data_for_signal = df_symbol_data.loc[date]
                            if isinstance(current_market_data_for_signal, pd.DataFrame):
                                # If duplicate index, use the first row for signal processing and price
                                current_market_data_for_signal = current_market_data_for_signal.iloc[0]
                            
                            # Pass historical data up to current date to strategy_fn
                            signal = strategy_fn(df_symbol_data.loc[:date]) 
                            self._process_signal(date, symbol, signal, current_market_data_for_signal)
                            current_price = current_market_data_for_signal['close']
                            daily_value += self.positions[symbol] * current_price
                        except IndexError: 
                            logger.warning(f"Data slice possibly empty for {symbol} on {date} for strategy_fn. No action taken.")
                            # Attempt to get current price for equity calculation even if no signal
                            if date in df_symbol_data.index:
                                current_price_data = df_symbol_data.loc[date]
                                if isinstance(current_price_data, pd.DataFrame):
                                    current_price_data = current_price_data.iloc[0]
                                current_price = current_price_data['close']
                                daily_value += self.positions[symbol] * current_price
                        except Exception as e:
                            logger.error(f"Error processing {symbol} on {date}: {e}")
                            continue
                equity_history.append({'date': date, 'equity': daily_value})

            self.equity_curve = pd.DataFrame(equity_history).set_index('date')['equity']
            return self._compute_metrics()

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise

    def _process_signal(
        self,
        date: datetime,
        symbol: str,
        signal: float,
        market_data: pd.Series
    ) -> None:
        price = market_data['close']
        available_capital = self.capital * self.config.position_size_limit

        if signal > 0 and self.capital >= price:
            max_shares = int(available_capital / price)
            shares = min(max_shares, 1)
            cost = shares * price * (1 + self.config.commission_rate + self.config.slippage_rate)
            if cost <= self.capital:
                self.positions[symbol] += shares
                self.capital -= cost
                self.trades.append(Trade(date, symbol, 'BUY', price, shares))
        elif signal < 0 and self.positions[symbol] > 0:
            shares = self.positions[symbol]
            proceeds = shares * price * (1 - self.config.commission_rate - self.config.slippage_rate)
            self.positions[symbol] = 0
            self.capital += proceeds
            self.trades.append(Trade(date, symbol, 'SELL', price, shares))

    def _compute_metrics(self) -> BacktestResults:
        returns = self.equity_curve.pct_change().dropna()
        if len(returns) == 0:
            raise ValueError("No trades executed during backtest period")

        profitable_trades = sum(
            1 for t in self.trades if t.side == 'SELL' and 
            any(bt.side == 'BUY' and bt.symbol == t.symbol and bt.price < t.price for bt in self.trades)
        )
        total_trades = len([t for t in self.trades if t.side == 'SELL'])

        return BacktestResults(
            total_return=(self.equity_curve.iloc[-1] / self.config.initial_capital) - 1,
            sharpe_ratio=self._calculate_sharpe_ratio(returns),
            max_drawdown=self._calculate_max_drawdown(),
            win_rate=profitable_trades / total_trades if total_trades > 0 else 0.0, # Ensure float
            profit_factor=self._calculate_profit_factor(),
            num_trades=total_trades,
            avg_trade_return=float(np.mean(returns)) if len(returns) > 0 else 0.0 # Ensure float
        )

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        excess_returns = returns - self.config.risk_free_rate / 252
        if excess_returns.std() == 0:
            return 0.0
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def _calculate_max_drawdown(self) -> float:
        peak = self.equity_curve.expanding(min_periods=1).max()
        drawdown = (self.equity_curve - peak) / peak
        return drawdown.min()

    def _calculate_profit_factor(self) -> float:
        """
        Calculate profit factor using FIFO pairing of BUY and SELL trades,
        properly accounting for trade quantities and prices.
        """
        buy_queue = []
        profits = []
        losses = []

        for t in self.trades:
            if t.side == 'BUY':
                buy_queue.append({'price': t.price, 'qty': t.quantity})
            elif t.side == 'SELL':
                qty_to_match = t.quantity
                sell_price = t.price
                while qty_to_match > 0 and buy_queue:
                    buy = buy_queue[0]
                    matched_qty = min(buy['qty'], qty_to_match)
                    pnl = (sell_price - buy['price']) * matched_qty
                    if pnl >= 0:
                        profits.append(pnl)
                    else:
                        losses.append(-pnl)
                    buy['qty'] -= matched_qty
                    qty_to_match -= matched_qty
                    if buy['qty'] == 0:
                        buy_queue.pop(0)

        total_profit = sum(profits)
        total_loss = sum(losses)
        if total_loss == 0:
            return float('inf') if total_profit > 0 else 0.0
        return total_profit / total_loss

    def export_results(self, output_dir: Path) -> None:
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.equity_curve.to_csv(output_dir / 'equity_curve.csv')
            trades_df = pd.DataFrame([t.__dict__ for t in self.trades])
            trades_df.to_csv(output_dir / 'trades.csv', index=False)
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            raise

# Strategy functions and lookup registry

def load_ohlcv(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Example OHLCV loader with validation using the central data validator.
    
    Args:
        symbol: Stock ticker symbol
        start: Start date for data range
        end: End date for data range
        
    Returns:
        DataFrame with validated OHLCV data
    """
    # Generate sample data
    dates = pd.date_range(start, end, freq='B')
    np.random.seed(hash(symbol) % 2**32)
    price = np.cumsum(np.random.randn(len(dates))) + 100
    # Define required_cols within the function scope
    required_cols_list = ['open', 'high', 'low', 'close', 'volume'] 
    df = pd.DataFrame({
        'open':  price + np.random.uniform(-1, 1, len(dates)),
        'high':  price + np.random.uniform(0, 2, len(dates)),
        'low':   price - np.random.uniform(0, 2, len(dates)),
        'close': price + np.random.uniform(-1, 1, len(dates)),
        'volume':np.random.randint(100000, 200000, len(dates))
    }, index=dates)
    df.index.name = 'date'
    
    # Validate the generated data using the centralized validator
    # required_cols = ['open', 'high', 'low', 'close', 'volume'] # This was defined locally
    validator = get_global_validator()
    result = validator.validate_dataframe(df, required_cols=required_cols_list, check_ohlcv=True, detect_anomalies_level='basic') # Use defined list
    
    if not result.is_valid:
        error_messages = result.errors if result.errors is not None else ["Unknown validation error"]
        logger.error(f"Generated data validation failed: {', '.join(error_messages)}")
        
    # Log statistics and warnings for monitoring
    if result.warnings:
        for warning in result.warnings:
            logger.warning(f"Data quality warning for {symbol}: {warning}")
    
    return df

def moving_average_crossover_strategy(df: pd.DataFrame, fast: int=10, slow: int=26) -> float:
    # existing implementation
    # Strategies should return a single float signal based on the input DataFrame slice
    # The DataFrame passed here is historical data up to the current simulation point.
    if df.empty:
        return 0.0
    # Example: simple signal if last close > 10-day SMA
    sma_fast = df['close'].rolling(window=min(fast, len(df))).mean()
    sma_slow = df['close'].rolling(window=min(slow, len(df))).mean()
    if sma_fast.empty or sma_slow.empty or len(sma_fast) < 1 or len(sma_slow) < 1:
        return 0.0
    if sma_fast.iloc[-1] > sma_slow.iloc[-1]:
        return 1.0 # Buy signal
    elif sma_fast.iloc[-1] < sma_slow.iloc[-1]:
        return -1.0 # Sell signal
    return 0.0 # Hold signal

def rsi_strategy(df: pd.DataFrame, period: int=14, overbought: float=70, oversold: float=30) -> float:
    if df.empty or len(df) < period:
        return 0.0
    
    try:
        rsi_series = calculate_rsi(df, length=period, close_col='close')
        if rsi_series.empty or pd.isna(rsi_series.iloc[-1]):
            return 0.0 # Not enough data or NaN result
        
        current_rsi = rsi_series.iloc[-1]
        
        if current_rsi > overbought:
            return -1.0  # Sell
        elif current_rsi < oversold:
            return 1.0   # Buy
        return 0.0   # Hold
    except Exception as e:
        logger.warning(f"RSI calculation failed in strategy: {e}")
        return 0.0

def macd_strategy(df: pd.DataFrame, fast: int=12, slow: int=26, signal_line: int=9) -> float:
    if df.empty or len(df) < slow + signal_line: # Ensure enough data for MACD calculation
        return 0.0
        
    try:
        macd_line_series, macd_signal_series, _ = calculate_macd(df, fast=fast, slow=slow, signal=signal_line, close_col='close')
        
        if macd_line_series.empty or macd_signal_series.empty or \
           pd.isna(macd_line_series.iloc[-1]) or pd.isna(macd_signal_series.iloc[-1]):
            return 0.0 # Not enough data or NaN result

        current_macd_line = macd_line_series.iloc[-1]
        current_macd_signal = macd_signal_series.iloc[-1]

        if current_macd_line > current_macd_signal:
            return 1.0  # Buy
        elif current_macd_line < current_macd_signal:
            return -1.0 # Sell
        return 0.0  # Hold
    except Exception as e:
        logger.warning(f"MACD calculation failed in strategy: {e}")
        return 0.0

STRATEGIES: Dict[str, Callable[[pd.DataFrame], float]] = { # Changed to float
    'Moving Average Crossover': moving_average_crossover_strategy,
    'SMA Crossover':            moving_average_crossover_strategy,
    'RSI Strategy':             rsi_strategy,
    'MACD Strategy':            macd_strategy,
}

def run_backtest(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    strategy: str,
    initial_capital: float,
    risk_per_trade: float,
    **kwargs  # <-- Accept extra strategy parameters
) -> Dict[str, Any]:
    """Run a backtest for a given trading strategy with improved safety."""
    # Validate dates using the modular data validator
    validator = get_global_validator()
    date_result = validator.validate_dates(
        start_date.date() if hasattr(start_date, 'date') else start_date, 
        end_date.date() if hasattr(end_date, 'date') else end_date
    )
    
    if not date_result.is_valid:
        error_messages = date_result.errors if date_result.errors is not None else ["Unknown date validation error"]
        logger.error(f"Invalid date range: {', '.join(error_messages)}")
        return {
            'metrics': empty_backtest_result(initial_capital)[0], 
            'equity_curve': empty_backtest_result(initial_capital)[1],
            'trade_log': empty_backtest_result(initial_capital)[2],
            'validation_errors': date_result.errors
        }
    
    # Load data after validation
    df = load_ohlcv(symbol, start_date, end_date)
    data = {symbol: df}

    config = BacktestConfig(
        initial_capital=initial_capital,
        commission_rate=0.001,
        slippage_rate=0.0005,
        risk_free_rate=0.02,
        position_size_limit=risk_per_trade
    )

    # Early data sufficiency check
    lookback = 0
    if strategy in ('Moving Average Crossover','SMA Crossover'):
        lookback = max(10,26)
    elif strategy == 'RSI Strategy':
        lookback = 14
    elif strategy == 'MACD Strategy':
        lookback = 26 + 9

    if len(df) < lookback:
        logger.warning(
            f"Insufficient data for {strategy} on {symbol}: "
            f"need {lookback}, got {len(df)}. Returning empty result."
        )
        metrics, equity_curve, trades = empty_backtest_result(initial_capital)
        return {'metrics': metrics, 'equity_curve': equity_curve, 'trade_log': trades}

    bt = Backtest(config)

    # Lookup strategy function
    strategy_fn = STRATEGIES.get(strategy)
    if not strategy_fn:
        logger.error(f"Unknown strategy: {strategy}")
        metrics, equity_curve, trades = empty_backtest_result(initial_capital)
        return {'metrics': metrics, 'equity_curve': equity_curve, 'trade_log': trades}

    # --- Generate signals using the selected strategy and kwargs ---
    try:
        # Pass kwargs to the strategy function if it accepts them
        import inspect
        sig = inspect.signature(strategy_fn)
        strategy_params = {k: v for k, v in kwargs.items() if k in sig.parameters}

        # The strategy_fn now directly returns a float signal when called with a data slice.
        # It will be called inside bt.simulate for each time step.
        # We need to create a partial function if kwargs are involved.
        from functools import partial
        if strategy_params:
            final_strategy_fn_for_simulate = partial(strategy_fn, **strategy_params)
        else:
            final_strategy_fn_for_simulate = strategy_fn

        results = bt.simulate(data, final_strategy_fn_for_simulate) 
        trades_df = pd.DataFrame([t.__dict__ for t in bt.trades])

    except ValueError as e:
        if 'No trades executed' in str(e):
            logger.warning("Strategy produced zero trades; returning empty result.")
            metrics, equity_curve, trades_df = empty_backtest_result(initial_capital)
            return {'metrics': metrics, 'equity_curve': equity_curve, 'trade_log': trades_df}
        raise

    equity_curve = bt.equity_curve.rename_axis("date").reset_index()
    metrics = results.dict()

    return {'metrics': metrics, 'equity_curve': equity_curve, 'trade_log': trades_df}

# Convenience wrapper for Streamlit pages

def run_backtest_wrapper(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    strategy: str,
    initial_capital: float,
    risk_per_trade: float,
    **kwargs  # <-- Accept extra strategy parameters
) -> Dict[str, Any]:
    try:
        return run_backtest(symbol, start_date, end_date, strategy, initial_capital, risk_per_trade, **kwargs)
    except Exception as e:
        logger.error(f"Backtest wrapper caught exception: {e}")
        metrics, equity_curve, trades = empty_backtest_result(initial_capital)
        return {'metrics': metrics, 'equity_curve': equity_curve, 'trade_log': trades}