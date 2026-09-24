"""
dashboard/components.py

Pure Streamlit rendering functions. No engine calls, no metric math --
that all lives in dashboard/backtest.py. Keeping this split means the
UI layer can be restyled without touching backtest logic, and vice versa.
"""

import pandas as pd
import streamlit as st


def render_top_metrics(result, closed_trades, open_trades, max_dd):
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Signals generated", result.total_signals_generated)
    col2.metric("Triggered", result.total_trades_triggered)
    col3.metric("Invalidated", result.total_invalidated)
    col4.metric("Expired", result.total_expired)
    col5.metric("Still open (end of data)", len(open_trades))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Win rate", f"{result.win_rate:.1f}%")
    col2.metric("Expectancy (R)", f"{result.expectancy:.2f}")
    col3.metric("Closed trades", len(closed_trades))
    col4.metric("Max drawdown (R)", f"{max_dd:.2f}")


def render_streaks(max_win_streak, max_loss_streak):
    col1, col2 = st.columns(2)
    col1.metric("Longest win streak", max_win_streak)
    col2.metric("Longest loss streak", max_loss_streak)


def render_equity_curve(equity_curve):
    st.subheader("Equity curve (cumulative R)")
    if equity_curve:
        st.line_chart(pd.DataFrame({"cumulative_R": equity_curve}))
    else:
        st.caption("No closed trades to chart yet.")


def render_long_short(long_stats, short_stats):
    st.subheader("Long vs short performance")
    lcol, scol = st.columns(2)
    with lcol:
        st.markdown("**LONG**")
        st.write(f"Trades: {long_stats['count']}")
        st.write(f"Win rate: {long_stats['win_rate']:.1f}%")
        st.write(f"Expectancy: {long_stats['expectancy']:.2f} R")
    with scol:
        st.markdown("**SHORT**")
        st.write(f"Trades: {short_stats['count']}")
        st.write(f"Win rate: {short_stats['win_rate']:.1f}%")
        st.write(f"Expectancy: {short_stats['expectancy']:.2f} R")


def render_trades_table(trades):
    st.subheader("Trades")
    if not trades:
        st.caption("No trades produced.")
        return

    rows = []
    for t in trades:
        rows.append(
            {
                "direction": t.direction.value if hasattr(t.direction, "value") else t.direction,
                "setup_time": t.setup_time,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "fill_price": t.fill_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "exit_price": t.exit_price,
                "result": t.result_status,
                "r_multiple": round(t.r_multiple, 3),
                "bars_held": t.bars_held,
            }
        )
    trades_df = pd.DataFrame(rows)
    st.dataframe(trades_df, use_container_width=True)
    st.download_button(
        "Download trades as CSV",
        trades_df.to_csv(index=False),
        file_name="backtest_trades.csv",
        mime="text/csv",
    )


def render_errors(errors):
    if not errors:
        return
    with st.expander(f"\u26a0\ufe0f {len(errors)} internal error(s) during replay", expanded=False):
        for err in errors:
            st.text(err)