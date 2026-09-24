from pathlib import Path

path = Path("strategy/liquidity.py")

with path.open("r", encoding="utf-8") as f:
    content = f.read()

NEW_SIG = """def update_liquidity_sweeps(
    df: pd.DataFrame,
    liquidity_levels: List[LiquidityLevel],
    resume_from: Optional[dict] = None,
) -> List[LiquidityLevel]:"""

OLD_SIG = """def update_liquidity_sweeps(
    df: pd.DataFrame,
    liquidity_levels: List[LiquidityLevel],
) -> List[LiquidityLevel]:"""

if content.count(NEW_SIG) != 1:
    raise RuntimeError("NEW_SIG anchor not found exactly once. No changes made.")

content = content.replace(NEW_SIG, OLD_SIG, 1)

NEW_BLOCK = """    for level in liquidity_levels:
        if level.status != LiquidityStatus.ACTIVE:
            continue

        # Stable identity for a liquidity level across repeated
        # detect_liquidity_levels() calls. The adapter may rebuild
        # LiquidityLevel objects, so object identity cannot be used.
        level_key = (
            level.liquidity_type,
            level.price,
            level.formed_at,
        )

        # The original causal boundary is strictly after formed_at.
        # A stored cursor may move that boundary forward only when the
        # previous scan proved that no sweep existed in the scanned range.
        lower_bound = level.formed_at

        if resume_from is not None:
            cursor = resume_from.get(level_key)

            if cursor is not None and cursor > lower_bound:
                lower_bound = cursor

        sub_df = df_copy[df_copy.index > lower_bound]

        if sub_df.empty:
            continue

        if level.liquidity_type == LiquidityType.SELL_SIDE:
            # SSL is swept if price drops below the swing low.
            mask = sub_df["low"] < level.price
        elif level.liquidity_type == LiquidityType.BUY_SIDE:
            # BSL is swept if price exceeds the swing high.
            mask = sub_df["high"] > level.price
        else:
            continue

        if not mask.any():
            # Nothing swept this level in the scanned range.
            # Remember the last scanned timestamp so a later call
            # can continue from here instead of rescanning history.
            if resume_from is not None:
                resume_from[level_key] = sub_df.index[-1]

            continue

        # First True preserves the original "first sweep wins" behavior.
        first_sweep_ts = mask.idxmax()

        level.status = LiquidityStatus.SWEPT
        level.sweep_bar_index = index_pos[first_sweep_ts]
        level.swept_at = first_sweep_ts

    return liquidity_levels"""

OLD_BLOCK = """    for level in liquidity_levels:
        if level.status != LiquidityStatus.ACTIVE:
            continue

        # Only bars strictly after the level was formed can sweep it --
        # a level can't be swept by the same bar (or an earlier bar) that
        # formed it.
        sub_df = df_copy[df_copy.index > level.formed_at]

        if sub_df.empty:
            continue

        if level.liquidity_type == LiquidityType.SELL_SIDE:
            # SSL is swept if price drops below the swing low.
            mask = sub_df["low"] < level.price
        elif level.liquidity_type == LiquidityType.BUY_SIDE:
            # BSL is swept if price exceeds the swing high.
            mask = sub_df["high"] > level.price
        else:
            continue

        if not mask.any():
            continue

        # First True in a boolean Series == first bar where the sweep
        # condition holds, same bar the original loop would have broken on.
        first_sweep_ts = mask.idxmax()

        level.status = LiquidityStatus.SWEPT
        level.sweep_bar_index = index_pos[first_sweep_ts]
        level.swept_at = first_sweep_ts

    return liquidity_levels"""

if content.count(NEW_BLOCK) != 1:
    raise RuntimeError("NEW_BLOCK anchor not found exactly once. No changes made.")

content = content.replace(NEW_BLOCK, OLD_BLOCK, 1)

with path.open("w", encoding="utf-8") as f:
    f.write(content)

print("PASS: liquidity.py reverted to pre-cursor state.")