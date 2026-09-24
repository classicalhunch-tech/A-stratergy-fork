from pathlib import Path

path = Path("phase_03_paper/tests/test_full_replay_integration.py")

with path.open("r", encoding="utf-8") as f:
    content = f.read()

OLD = 'DATA_FILE = "your_data_file.csv"'
NEW = 'DATA_FILE = "your_data_file_500.csv"'

if content.count(OLD) != 1:
    raise RuntimeError("Anchor not found exactly once. No changes made.")

content = content.replace(OLD, NEW, 1)

with path.open("w", encoding="utf-8") as f:
    f.write(content)

print("PASS: test_full_replay_integration.py now points at your_data_file_500.csv")