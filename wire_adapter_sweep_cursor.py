from pathlib import Path

path = Path("phase_03_paper/signals/adapter.py")

with path.open("r", encoding="utf-8") as f:
    content = f.read()

OLD_INIT = """        self._prepared_context = None

        # --------------------------------------------------------
        # Persistent cross-candle strategy state
        # --------------------------------------------------------"""

NEW_INIT = """        self._prepared_context = None

        # --------------------------------------------------------
        # Persistent cursor for update_liquidity_sweeps(), so bars
        # already scanned (and found not to sweep a level) are not
        # rescanned on every subsequent candle. Keyed naturally by
        # (liquidity_type, price, formed_at) since liquidity levels
        # are rebuilt fresh each candle by detect_liquidity_levels().
        # --------------------------------------------------------

        self._sweep_resume_cursors: dict = {}

        # --------------------------------------------------------
        # Persistent cross-candle strategy state
        # --------------------------------------------------------"""

if content.count(OLD_INIT) != 1:
    raise RuntimeError("INIT anchor not found exactly once. No changes made.")

content = content.replace(OLD_INIT, NEW_INIT, 1)

OLD_CALL = """        all_liquidity_levels = update_liquidity_sweeps(
            df,
            all_liquidity_levels,
        )"""

NEW_CALL = """        all_liquidity_levels = update_liquidity_sweeps(
            df,
            all_liquidity_levels,
            resume_from=self._sweep_resume_cursors,
        )"""

if content.count(OLD_CALL) != 1:
    raise RuntimeError("CALL anchor not found exactly once. No changes made.")

content = content.replace(OLD_CALL, NEW_CALL, 1)

with path.open("w", encoding="utf-8") as f:
    f.write(content)

print("PASS: adapter.py now passes a persistent resume cursor to update_liquidity_sweeps().")