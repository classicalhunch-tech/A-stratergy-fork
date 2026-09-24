"""
phase_02_optimization/overfit_sanity_no_confounds.py

PBO sanity check with known-confounded trials excluded.

Primary PBO:
    - 15 trials
    - CSCV_BLOCKS = 10
    - PBO calculated from the full trial universe

Sanity check:
    - Removes trials marked as known-confounded
      by trial_registry.py.
    - Currently this removes only:
          entry_mode='extreme'

Why remove it?
    The extreme-entry trial is known to be confounded because
    the current signal implementation silently forces an
    effective stop_buffer=0.01 for that mode.

    It also produced only 2 closed trades, while the other
    trials produced approximately 14-17 closed trades.

Purpose:
    Determine whether the very high primary PBO result is
    materially affected by this known-confounded trial.

Safety:
    - Does NOT modify strategy logic.
    - Does NOT modify the canonical backtest.
    - Does NOT modify trial_registry.py on disk.
    - Does NOT modify overfit_detection.py on disk.
    - Keeps CSCV_BLOCKS unchanged at the primary value of 10.
    - Uses the existing PBO cache.
    - Does not intentionally create a new cache.
    - Restores the temporary function override before exiting.

Usage:
    python -m phase_02_optimization.overfit_sanity_no_confounds
"""

import phase_02_optimization.overfit_detection as od
import phase_02_optimization.trial_registry as tr


def filtered_build_trial_configs():
    """
    Return the normal trial universe with all known-confounded
    trials removed.
    """

    return [
        (config, source_axis, changed_parameter, confound)
        for config, source_axis, changed_parameter, confound
        in tr.build_trial_configs()
        if confound is None
    ]


def main() -> None:
    """
    Run the PBO sanity check using only non-confounded trials.
    """

    # Save the original function so the imported module can be
    # restored before this process exits.
    original_build_trial_configs = od.build_trial_configs

    try:
        # Temporarily replace the trial-universe builder used by
        # overfit_detection.py.
        #
        # This affects only this Python process.
        # It does NOT modify the source file on disk.
        od.build_trial_configs = filtered_build_trial_configs

        print()
        print("=" * 100)
        print("PBO SANITY CHECK -- KNOWN CONFOUNDED TRIAL EXCLUDED")
        print("=" * 100)

        print()
        print("Primary PBO universe : 15 trials")
        print("Sanity-check universe: 14 non-confounded trials")
        print(f"CSCV block count     : {od.CSCV_BLOCKS}")
        print(f"Cache                : {od.PBO_CACHE_JSON}")

        print()
        print("Excluded known-confounded trial:")
        print("  entry_mode = 'extreme'")

        print()
        print("=" * 100)
        print("LOADING DATASET")
        print("=" * 100)

        df = od.load_dataset(od.DATA_FILE)

        print(
            f"Dataset candles loaded : {len(df)}"
        )

        ts_index = od.build_timestamp_index(df)

        print()
        print("=" * 100)
        print("BUILDING FILTERED PERFORMANCE MATRIX")
        print("=" * 100)

        labels, matrix = od.build_block_performance_matrix(
            df,
            ts_index,
        )

        print()
        print(
            f"Trials in sanity universe : {len(matrix)}"
        )

        if len(matrix) != 14:
            raise RuntimeError(
                "Expected exactly 14 non-confounded trials, "
                f"but found {len(matrix)}."
            )

        print(
            "Expected non-confounded trials: 14"
        )

        print()
        print("=" * 100)
        print("RUNNING FILTERED PBO")
        print("=" * 100)

        od.run_pbo(labels, matrix)

        print()
        print("=" * 100)
        print("PBO NO-CONFOUND SANITY CHECK COMPLETE")
        print("=" * 100)

        print()
        print(
            "Primary PBO cache was used:"
        )

        print(
            f"  {od.PBO_CACHE_JSON}"
        )

        print()
        print(
            "The strategy, canonical backtest, "
            "trial_registry.py, and overfit_detection.py "
            "were not modified."
        )

        print()

    finally:
        # Restore the original function even if the sanity check
        # raises an exception.
        od.build_trial_configs = original_build_trial_configs


if __name__ == "__main__":
    main()