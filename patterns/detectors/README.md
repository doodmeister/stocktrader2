# Candlestick Pattern Detectors

This directory contains the implementation of candlestick pattern detection algorithms used by the StockTrader bot for technical analysis and trading signal generation.

## 📁 Directory Structure

```
patterns/detectors/
├── __init__.py                    # Pattern detector exports and imports
├── README.md                      # This documentation file
├── hammer.py                      # Hammer pattern (fundamental reversal)
├── doji.py                        # Doji pattern (indecision/reversal)
├── engulfing.py                   # Bullish Engulfing pattern (reversal)
├── morning_star.py                # Morning Star pattern (3-candle reversal)
├── bullish_patterns.py            # Collection of 9 bullish reversal patterns
└── bearish_patterns.py            # Collection of 5 bearish reversal patterns
```

## 🏗️ Architecture Design

### Hybrid Organization Approach

We use a **hybrid organization** that balances maintainability with functionality:

#### Individual Pattern Files
The most fundamental and commonly used patterns get their own dedicated files:
- **`hammer.py`** (85 lines) - Critical bullish reversal pattern
- **`doji.py`** (76 lines) - Key indecision/reversal pattern  
- **`engulfing.py`** (72 lines) - Very common reversal pattern
- **`morning_star.py`** (92 lines) - Important 3-candle reversal pattern

#### Grouped Pattern Files
Related patterns are logically grouped to reduce file proliferation:
- **`bullish_patterns.py`** (462 lines) - 9 bullish reversal patterns
- **`bearish_patterns.py`** (262 lines) - 5 bearish reversal patterns

### Why This Organization?

✅ **Advantages:**
- **Focused Individual Files**: Most important patterns get dedicated attention
- **Logical Grouping**: Related patterns share validation logic and utilities
- **Manageable File Count**: Avoids 18+ individual files cluttering the directory
- **Import Simplicity**: Easy to import related patterns together
- **Maintenance Efficiency**: Similar patterns maintained in one place
- **Performance**: Grouped imports are more efficient

❌ **Alternative Approaches Considered:**
- **All Individual Files**: Would create too many files (18+)
- **All Grouped Files**: Would lose focus on critical patterns
- **By Candle Count**: Less intuitive for trading logic
- **By Market Context**: More complex organization

## 📊 Pattern Inventory

### Individual Patterns (4 patterns)
| Pattern | File | Type | Candles | Description |
|---------|------|------|---------|-------------|
| Hammer | `hammer.py` | Bullish Reversal | 1 | Small body, long lower shadow |
| Doji | `doji.py` | Indecision | 1 | Open ≈ Close, market indecision |
| Bullish Engulfing | `engulfing.py` | Bullish Reversal | 2 | Large bullish candle engulfs bearish |
| Morning Star | `morning_star.py` | Bullish Reversal | 3 | Gap down, doji/small, gap up |

### Bullish Patterns (9 patterns in `bullish_patterns.py`)
| Pattern | Type | Candles | Description |
|---------|------|---------|-------------|
| Piercing Pattern | Bullish Reversal | 2 | Opens below, closes above midpoint |
| Bullish Harami | Bullish Reversal | 2 | Small bullish inside large bearish |
| Three White Soldiers | Bullish Reversal | 3 | Three consecutive bullish candles |
| Inverted Hammer | Bullish Reversal | 1 | Small body, long upper shadow |
| Morning Doji Star | Bullish Reversal | 3 | Gap down doji, gap up bullish |
| Bullish Abandoned Baby | Bullish Reversal | 3 | Rare gap reversal pattern |
| Bullish Belt Hold | Bullish Reversal | 1 | Opens at low, closes near high |
| Three Inside Up | Bullish Reversal | 3 | Harami followed by confirmation |
| Rising Window | Bullish Continuation | 2 | Gap up continuation pattern |

### Bearish Patterns (5 patterns in `bearish_patterns.py`)
| Pattern | Type | Candles | Description |
|---------|------|---------|-------------|
| Bearish Engulfing | Bearish Reversal | 2 | Large bearish candle engulfs bullish |
| Evening Star | Bearish Reversal | 3 | Gap up, doji/small, gap down |
| Three Black Crows | Bearish Reversal | 3 | Three consecutive bearish candles |
| Bearish Harami | Bearish Reversal | 2 | Small bearish inside large bullish |
| Upside Gap Two Crows | Bearish Reversal | 3 | Gap up followed by two bearish |

## 🔧 Implementation Standards

### Base Class Integration
All patterns inherit from `PatternDetector` base class:
```python
from ..base import PatternDetector, PatternResult, PatternType, PatternStrength
```

### Pattern Structure
Each pattern implements:
- **`__init__()`**: Pattern configuration (name, type, min_rows)
- **`detect()`**: Main detection logic with validation
- **`_determine_strength()`**: Confidence-based strength calculation
- **Validation**: Uses `@validate_dataframe` decorator

### Code Quality Standards
- **Type Hints**: Full type annotations for all parameters
- **Error Handling**: Graceful handling of edge cases
- **Documentation**: Comprehensive docstrings with pattern descriptions
- **Validation**: Input validation and data integrity checks
- **Confidence Scoring**: 0.0-1.0 confidence levels for detection
- **Pattern Strength**: Weak/Medium/Strong strength classification

## 🚀 Usage Examples

### Individual Pattern Usage
```python
from patterns.detectors import HammerPattern, DojiPattern

# Create pattern detectors
hammer = HammerPattern()
doji = DojiPattern()

# Detect patterns in OHLCV data
hammer_result = hammer.detect(ohlcv_df)
doji_result = doji.detect(ohlcv_df)

# Check results
if hammer_result.detected:
    print(f"Hammer detected with {hammer_result.confidence:.2f} confidence")
```

### Orchestrated Pattern Detection
```python
from patterns.orchestrator import CandlestickPatterns

# Create orchestrator
detector = CandlestickPatterns(confidence_threshold=0.7)

# Detect all patterns
results = detector.detect_all_patterns(ohlcv_df)

# Filter by type
bullish_patterns = detector.get_bullish_patterns(ohlcv_df)
bearish_patterns = detector.get_bearish_patterns(ohlcv_df)
```

### Factory Pattern Creation
```python
from patterns.factory import create_pattern_detector

# Create detector with default configuration
detector = create_pattern_detector()

# Detect patterns
all_results = detector.detect_all_patterns(ohlcv_df)
```

## 🔄 Integration with StockTrader Bot

### Core Integration Points
1. **Feature Engineering**: Patterns used in `train/feature_engineering.py`
2. **ML Training**: Pattern labels for `train/deeplearning_trainer.py`
3. **Risk Management**: Pattern signals in trading decisions
4. **API Endpoints**: Real-time pattern detection via FastAPI
5. **WebSocket Streaming**: Live pattern alerts

### Data Flow
```
OHLCV Data → Pattern Detection → Feature Engineering → ML Training
                ↓                      ↓
        Real-time Alerts    →    Trading Signals
```

## 📈 Performance Considerations

### Optimization Features
- **Parallel Processing**: Optional multi-threading support
- **Caching**: Pattern result caching in orchestrator
- **Efficient Validation**: Minimal data copying
- **Batch Processing**: Multiple symbol support

### Memory Management
- **Lazy Loading**: Patterns loaded on-demand
- **Data Efficiency**: Minimal memory footprint
- **Result Caching**: Configurable cache size limits

## 🧪 Testing and Validation

### Test Coverage
Each pattern includes:
- **Unit Tests**: Individual pattern logic testing
- **Integration Tests**: Orchestrator functionality
- **Performance Tests**: Speed and memory benchmarks
- **Data Validation**: Edge case handling

### Quality Assurance
- **Type Safety**: MyPy static type checking
- **Code Quality**: Ruff linting and formatting
- **Documentation**: Comprehensive docstring coverage
- **Error Handling**: Robust exception management

## 🔮 Future Enhancements

### Planned Additions
- **Additional Patterns**: More candlestick patterns
- **ML Enhancement**: AI-powered pattern recognition
- **Performance Optimization**: Cython/Numba acceleration
- **Custom Patterns**: User-defined pattern support

### Architecture Evolution
- **Plugin System**: Dynamic pattern loading
- **Configuration**: Runtime pattern customization
- **Metrics**: Advanced performance monitoring
- **Visualization**: Pattern detection charts

## 📝 Contributing

### Adding New Patterns
1. **Individual Pattern**: Create new `.py` file for critical patterns
2. **Grouped Pattern**: Add to appropriate `*_patterns.py` file
3. **Update Exports**: Add to `__init__.py` imports
4. **Documentation**: Update this README.md
5. **Tests**: Add comprehensive test coverage

### Code Standards
- Follow existing pattern structure and naming
- Include comprehensive docstrings
- Add type hints for all parameters
- Implement proper error handling
- Include confidence scoring logic

---

*For more information, see the main project documentation in the root `README.md` file.*
