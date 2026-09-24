import MetaTrader5 as mt5

if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
else:
    account_info = mt5.account_info()
    if account_info:
        print(f"Connected to MT5 Demo! Account: {account_info.login}")
    else:
        print("initialize() succeeded but account_info() returned None")
    mt5.shutdown()