"""
market_data/research/ -- descriptive research tools only.

Nothing in this package is causal-safe for live use by itself:
future_returns.py deliberately looks forward (it produces the outcome
label, not an input feature). Only market_data/features.py is meant to
ever be wired into anything that resembles a live decision path.
"""