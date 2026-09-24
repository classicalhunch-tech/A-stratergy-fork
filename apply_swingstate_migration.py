"""
apply_swingstate_migration.py

One-time migration:

    Moves the SwingState class OUT of
    phase_02_optimization/test_swing_state_equivalence.py
    and INTO strategy/swings.py (appended after find_swings(),
    which is never touched).

Why:
    The test file currently contains its own full duplicate copy
    of find_swings() alongside the new SwingState class. That
    means the "equivalence test" would just be comparing code
    against an identical copy of itself -- it proves nothing, and
    strategy/ is supposed to be the single source of truth. The two
    copies have already silently drifted by one character (an
    error-message typo fix in the duplicate) -- exactly the failure
    mode this migration removes.

Safety:
    - Refuses to run twice (checks strategy/swings.py doesn't
      already contain SwingState).
    - Writes .bak backups of both files before changing anything.
    - Never touches a single character of find_swings() itself --
      it locates the end of that function textually and only
      copies/moves content that comes AFTER it.

Revert:
    Restore the two .bak files:

        Copy-Item .\\strategy\\swings.py.bak .\\strategy\\swings.py -Force
        Copy-Item .\\phase_02_optimization\\test_swing_state_equivalence.py.bak `
            .\\phase_02_optimization\\test_swing_state_equivalence.py -Force

Run:
    python .\\apply_swingstate_migration.py
"""

from pathlib import Path

SWINGS_PATH = Path("strategy/swings.py")
TEST_PATH = Path("phase_02_optimization/test_swing_state_equivalence.py")

FIND_SWINGS_END_MARKER = "    return swings"

IMPORT_LINE = "from enum import Enum"
TYPING_IMPORT = "from typing import Optional"


def fail(message):
    print(f"❌ ERROR: {message}")
    raise SystemExit(1)


print()
print("=" * 70)
print("🔧 SWINGSTATE MIGRATION — move SwingState into strategy/swings.py")
print("=" * 70)
print()

if not SWINGS_PATH.exists():
    fail(f"{SWINGS_PATH} not found. Run this from the project root.")

if not TEST_PATH.exists():
    fail(f"{TEST_PATH} not found. Run this from the project root.")

swings_src = SWINGS_PATH.read_text(encoding="utf-8")
test_src = TEST_PATH.read_text(encoding="utf-8")

if "class SwingState" in swings_src:
    fail(
        f"{SWINGS_PATH} already contains SwingState. "
        "Migration already applied — refusing to run twice."
    )

if "class SwingState" not in test_src:
    fail(
        f"{TEST_PATH} does not contain a SwingState class. "
        "Nothing to migrate."
    )

# ------------------------------------------------------------------
# Locate the end of find_swings() inside the TEST file's duplicate
# copy, so we can grab everything AFTER it (the SwingState section)
# without touching find_swings() itself anywhere.
# ------------------------------------------------------------------

marker_index = test_src.find(FIND_SWINGS_END_MARKER)

if marker_index == -1:
    fail(
        "Could not find the end of find_swings() inside the test "
        "file (looked for the line 'return swings'). Aborting "
        "without changing anything — the file may have been edited "
        "since this script was written."
    )

# Take everything after the LAST 'return swings' occurrence, in case
# the string appears more than once for any reason.
marker_index = test_src.rfind(FIND_SWINGS_END_MARKER)
swingstate_section = test_src[marker_index + len(FIND_SWINGS_END_MARKER):]
swingstate_section = swingstate_section.lstrip("\n")

if "class SwingState" not in swingstate_section:
    fail(
        "Extracted section after find_swings() does not contain "
        "SwingState. Aborting without changing anything."
    )

print("✅ Located SwingState section in the test file.")
print(f"   Extracted {len(swingstate_section):,} characters.")

# ------------------------------------------------------------------
# BACK UP both files before touching anything.
# ------------------------------------------------------------------

swings_backup = SWINGS_PATH.with_suffix(".py.bak")
test_backup = TEST_PATH.with_suffix(".py.bak")

swings_backup.write_text(swings_src, encoding="utf-8")
test_backup.write_text(test_src, encoding="utf-8")

print(f"💾 Backed up: {swings_backup}")
print(f"💾 Backed up: {test_backup}")

# ------------------------------------------------------------------
# Build the new strategy/swings.py: original content, byte-for-byte,
# plus the typing import, plus the SwingState section appended.
# ------------------------------------------------------------------

new_swings_src = swings_src

if TYPING_IMPORT not in new_swings_src:
    if IMPORT_LINE not in new_swings_src:
        fail(
            f"Expected to find the line {IMPORT_LINE!r} in "
            f"{SWINGS_PATH} to insert the typing import next to it, "
            "but it wasn't there. Aborting without changing anything."
        )

    new_swings_src = new_swings_src.replace(
        IMPORT_LINE,
        f"{IMPORT_LINE}\n{TYPING_IMPORT}",
        1,
    )

    print(f"✅ Added '{TYPING_IMPORT}' to the imports.")

if not new_swings_src.endswith("\n"):
    new_swings_src += "\n"

new_swings_src += "\n\n" + swingstate_section

SWINGS_PATH.write_text(new_swings_src, encoding="utf-8")

print(f"✅ Appended SwingState to {SWINGS_PATH}.")

# ------------------------------------------------------------------
# Strip the duplicate find_swings()/SwingState out of the test file,
# leaving a clean stub that imports both from strategy.swings. The
# real test bodies are written in the NEXT step, once the CSV
# loading helper (phase_03_paper/replay/csv_source.py) is reviewed.
# ------------------------------------------------------------------

stub_test_src = '''"""
phase_02_optimization/test_swing_state_equivalence.py

Proves SwingState.step() (called once per candle, in chronological
order) produces EXACTLY the same result as find_swings(df) (called
once on the full historical DataFrame).

Both are imported from strategy.swings -- the single source of
truth. This file must never contain its own copy of either.

STATUS: stub after the SwingState migration. Test bodies are added
next, once the CSV-loading approach is finalized (reusing the
already-tested phase_03_paper/replay/csv_source.py loader rather
than writing a fourth CSV parser).
"""

from strategy.swings import find_swings, SwingState  # noqa: F401
'''

TEST_PATH.write_text(stub_test_src, encoding="utf-8")

print(f"✅ Rewrote {TEST_PATH} as a clean stub (no duplicated logic).")

print()
print("=" * 70)
print("🏁 MIGRATION COMPLETE")
print("=" * 70)
print()
print("Next steps:")
print("  1. python -m py_compile .\\strategy\\swings.py")
print("  2. python -m py_compile .\\phase_02_optimization\\test_swing_state_equivalence.py")
print("  3. python -m unittest discover -s tests -v        (expect 79 passing)")
print("  4. python -m unittest discover -s phase_03_paper -v  (expect 82 passing)")
print()