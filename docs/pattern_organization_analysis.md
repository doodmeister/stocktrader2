"""
Recommended Pattern Organization Structure
"""

# Current structure is good, but could be optimized:

patterns/detectors/
├── __init__.py                    # Import aggregation
├── basic_patterns.py              # Simple 1-candle patterns (hammer, doji, etc.)
├── two_candle_patterns.py         # 2-candle patterns (engulfing, piercing, etc.)
├── three_candle_patterns.py       # 3-candle patterns (morning star, evening star, etc.)
├── complex_patterns.py            # Multi-candle complex patterns
└── specialized_patterns.py        # Special cases and rare patterns

# Alternative organization by type:
patterns/detectors/
├── __init__.py
├── reversal_patterns.py           # All reversal patterns
├── continuation_patterns.py       # Continuation patterns
├── indecision_patterns.py         # Doji and spinning tops
└── gap_patterns.py                # Gap-based patterns

# Current hybrid approach (RECOMMENDED):
patterns/detectors/
├── __init__.py                    # ✓ Good aggregation
├── hammer.py                      # ✓ Keep - fundamental pattern
├── doji.py                        # ✓ Keep - fundamental pattern  
├── engulfing.py                   # ✓ Keep - very common pattern
├── morning_star.py                # ✓ Keep - important 3-candle pattern
├── bullish_patterns.py            # ✓ Keep - groups related patterns well
└── bearish_patterns.py            # ✓ Keep - groups related patterns well
