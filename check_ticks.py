import MetaTrader5 as mt5
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

mt5.initialize()
mt5.symbol_select("XAUUSD", True)

flag_counts = Counter()
last_nonzero = 0
n = 0

end = time.time() + 60
prev_time_msc = None
while time.time() < end:
    t = mt5.symbol_info_tick("XAUUSD")
    if t and t.time_msc != prev_time_msc:
        prev_time_msc = t.time_msc
        n += 1
        flag_counts[t.flags] += 1
        if t.last != 0.0:
            last_nonzero += 1
    time.sleep(0.1)

print("sampled", n, "distinct ticks over 60s")
print("flag value counts:", dict(flag_counts))
print("ticks with nonzero last:", last_nonzero)

utc_to = datetime.now(timezone.utc)
utc_from = utc_to - timedelta(hours=2)
ticks = mt5.copy_ticks_range("XAUUSD", utc_from, utc_to, mt5.COPY_TICKS_ALL)
print("historical ticks now:", 0 if ticks is None else len(ticks))

mt5.shutdown()
