from pathlib import Path

path = Path("phase_03_paper/tests/test_event_collector_smoke.py")

with path.open("r", encoding="utf-8") as f:
    content = f.read()

OLD = "        self.assertEqual(summary.total_candles, 5000)\n        self.assertEqual(summary.ticks_run, 5000)"
NEW = "        self.assertEqual(summary.total_candles, 500)\n        self.assertEqual(summary.ticks_run, 500)"

if content.count(OLD) != 1:
    raise RuntimeError("Anchor not found exactly once. No changes made.")

content = content.replace(OLD, NEW, 1)

content = content.replace(
    "Confirms the event collector, run against the SAME real 5000-candle",
    "Confirms the event collector, run against the SAME real 500-candle",
    1,
)

with path.open("w", encoding="utf-8") as f:
    f.write(content)

print("PASS: test reverted to expect 500 candles.")