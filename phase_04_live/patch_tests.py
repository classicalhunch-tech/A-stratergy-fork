"""
patch_tests.py

One-time patch for phase_04_live/orders/test_order_manager.py.

Adds a mocked check_symbol_compatibility() so the existing tests (which
predate the Step 3 compatibility gate in order_manager.py) don't hit the
real MT5 symbol_info() call and fail with "No IPC connection".

Run from the project root:

    python patch_tests.py

It is idempotent -- running it twice will not double-patch the file.
"""

from pathlib import Path

TARGET = Path("phase_04_live/orders/test_order_manager.py")

OLD_HELPER_ANCHOR = '''def make_broker_result(retcode: int, order: int = 123456, comment: str = ""):
    """A stand-in for the object mt5.order_send() returns."""
    return SimpleNamespace(retcode=retcode, order=order, comment=comment)
'''

NEW_HELPER_BLOCK = '''def make_broker_result(retcode: int, order: int = 123456, comment: str = ""):
    """A stand-in for the object mt5.order_send() returns."""
    return SimpleNamespace(retcode=retcode, order=order, comment=comment)


def make_compatibility_report(
    *,
    is_compatible: bool = True,
    resolved_order_filling_mode=mt5.ORDER_FILLING_IOC,
    blocking_issues=None,
):
    """A stand-in for check_symbol_compatibility()'s return value.

    order_manager.py only ever reads .is_compatible and
    .resolved_order_filling_mode (plus .blocking_issues on the
    rejection path), so a duck-typed SimpleNamespace is a faithful,
    low-coupling test double -- same rationale as make_tick/make_signal
    above. This avoids coupling the test file to
    SymbolCompatibilityReport's exact constructor signature.
    """
    return SimpleNamespace(
        is_compatible=is_compatible,
        resolved_order_filling_mode=resolved_order_filling_mode,
        blocking_issues=blocking_issues or [],
    )
'''

OLD_SETUP = '''        self.config = make_config(dry_run=True)

        # Tight, clean spread around 1.10000 -- well within max_spread_points.
        self.tick = make_tick(bid=1.09999, ask=1.10001)
'''

NEW_SETUP = '''        self.config = make_config(dry_run=True)

        # Tight, clean spread around 1.10000 -- well within max_spread_points.
        self.tick = make_tick(bid=1.09999, ask=1.10001)

        # The Step 3 compatibility gate (_resolve_filling_mode) runs
        # before any tick is fetched or order is sent. Patch it here,
        # once, for every test in every subclass, so existing tests
        # written before that gate existed don't hit the real MT5
        # symbol_info() call. Individual tests can still override
        # self.mock_compat.return_value to exercise the rejection path.
        compat_patcher = patch(
            f"{ORDER_MANAGER_MODULE}.check_symbol_compatibility"
        )
        self.mock_compat = compat_patcher.start()
        self.mock_compat.return_value = make_compatibility_report()
        self.addCleanup(compat_patcher.stop)
'''


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"Could not find {TARGET} -- run this from the project root.")

    text = TARGET.read_text(encoding="utf-8")

    already_patched = "make_compatibility_report" in text
    if already_patched:
        print("Already patched (make_compatibility_report found). Nothing to do.")
        return

    if OLD_HELPER_ANCHOR not in text:
        raise SystemExit(
            "Could not find make_broker_result() anchor text. "
            "The file may have changed since this patch was written -- "
            "apply the edit manually."
        )

    if OLD_SETUP not in text:
        raise SystemExit(
            "Could not find setUp() anchor text. "
            "The file may have changed since this patch was written -- "
            "apply the edit manually."
        )

    text = text.replace(OLD_HELPER_ANCHOR, NEW_HELPER_BLOCK, 1)
    text = text.replace(OLD_SETUP, NEW_SETUP, 1)

    TARGET.write_text(text, encoding="utf-8")
    print(f"Patched {TARGET}")


if __name__ == "__main__":
    main()