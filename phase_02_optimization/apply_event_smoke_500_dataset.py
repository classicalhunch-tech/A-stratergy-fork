from pathlib import Path

path = Path("phase_03_paper/tests/test_event_collector_smoke.py")

with path.open("r", encoding="utf-8") as f:
    content = f.read()

OLD = 'Path(__file__).resolve().parents[2] / "your_data_file.csv"'
NEW = 'Path(__file__).resolve().parents[2] / "your_data_file_500.csv"'

if content.count(OLD) != 1:
    raise RuntimeError("Anchor not found exactly once. No changes made.")

content = content.replace(OLD, NEW, 1)

with path.open("w", encoding="utf-8") as f:
    f.write(content)

print("PASS: test_event_collector_smoke.py now points at your_data_file_500.csv")