"""
market_data/ -- market-agnostic microstructure data layer.

This package is intentionally independent of `strategy/` and of the
OHLCV-proxy `orderflow/` layer. It normalizes raw exchange data (starting
with Binance spot, used as a research laboratory per project rules) into
a common event schema:

    TRADE event -- actual aggressor-side trades
    BOOK  event -- top-of-book liquidity state

Nothing here talks to XAUUSD, MT5, or the strategy package. It exists to
build and validate order-flow *features* on a market with richer public
data, before those features are ever considered for the target market.
"""