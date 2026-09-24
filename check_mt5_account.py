import MetaTrader5 as mt5

# Initialize the connection to your running MT5 terminal
if not mt5.initialize():
    print("Initialization failed, error code =", mt5.last_error())
else:
    # Verify by printing your demo account info
    acc = mt5.account_info()
    print(f"Successfully connected to account: {acc.login} on {acc.server}")
    print(f"Demo Balance: {acc.balance}")