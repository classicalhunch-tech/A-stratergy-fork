"""
gold_orderflow/ -- investigating whether genuine (non-proxy) order-flow
information is available for XAUUSD via your existing MT5 connection.

Separate from market_data/ (the Binance crypto lab) and from orderflow/
(the OHLCV proxy layer on gold). This package exists purely to find out
what real data is actually available before building any feature on it --
same discipline as everywhere else in this project: inspect first, don't
assume.
"""