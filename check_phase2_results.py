import json

with open("phase2_dashboard/phase2_results.json", encoding="utf-8") as f:
    data = json.load(f)

for k, v in data.items():
    if isinstance(v, list) and v:
        last = v[-1]
        rc = last.get("returncode")
        ok = last.get("succeeded")
        n = len(v)
        print(f"{k:25s} entries={n:3d}  returncode={rc}  succeeded={ok}")
    else:
        print(f"{k:25s} entries=0 (EMPTY)")