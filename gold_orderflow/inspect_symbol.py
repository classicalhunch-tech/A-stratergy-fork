"""
gold_orderflow/inspect_symbol.py

Diagnostic-only. Inspects XAUUSD symbol metadata and current market state
via MT5. Does NOT place orders, does NOT pull historical data.
"""

import argparse
import MetaTrader5 as mt5


def inspect_symbol(symbol: str) -> None:
    if not mt5.initialize():
        print("mt5.initialize() failed:", mt5.last_error())
        return

    print(f"terminal info: {mt5.terminal_info()}")
    print(f"account info: {mt5.account_info()}")
    print("-" * 70)

    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"symbol_info('{symbol}') returned None -> symbol not found")
        mt5.shutdown()
        return

    print(f"name:            {info.name}")
    print(f"visible:         {info.visible}")
    print(f"trade_mode:      {info.trade_mode}  "
          f"(0=disabled,1=longonly,2=shortonly,3=closeonly,4=full)")
    print(f"point:           {info.point}")
    print(f"digits:          {info.digits}")
    print(f"spread:          {info.spread}")
    print(f"spread_float:    {info.spread_float}")
    print(f"bid:             {info.bid}")
    print(f"ask:             {info.ask}")
    print(f"last:            {info.last}")
    print(f"volume:          {info.volume}")
    print(f"volumehigh:      {info.volumehigh}")
    print(f"volumelow:       {info.volumelow}")
    print(f"time:            {info.time}")
    print(f"path:            {info.path}")
    print(f"description:     {info.description}")

    print("-" * 70)
    tick = mt5.symbol_info_tick(symbol)
    print(f"symbol_info_tick('{symbol}'): {tick}")

    if tick is not None:
        flags = tick.flags
        bit_names = {
            2: "BID", 4: "ASK", 8: "LAST",
            16: "VOLUME", 32: "BUY", 64: "SELL",
        }
        present = [name for bit, name in bit_names.items() if flags & bit]
        print(f"decoded flags ({flags}): {present}")

    mt5.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    args = parser.parse_args()
    inspect_symbol(args.symbol)