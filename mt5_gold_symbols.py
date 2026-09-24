import MetaTrader5 as mt5

if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
else:
    symbols = mt5.symbols_get()
    gold_like = [s.name for s in symbols if "XAU" in s.name.upper() or "GOLD" in s.name.upper()]
    print("Gold-related symbols on this account:")
    for name in gold_like:
        print(" -", name)
    mt5.shutdown()