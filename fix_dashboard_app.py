"""
fix_dashboard_app.py

One-time script to update dashboard/app.py:
  1. Flip Phase 2's status from LOCKED to ACTIVE.
  2. Replace the disabled "Phase 2 Locked" button with a working
     st.page_link to the new Phase 2 Lab page.

Run this from the project root:

    python fix_dashboard_app.py

It is safe to run more than once — if the target blocks are not
found (e.g. because the edit already applied), it will say so
and make no changes.
"""

from pathlib import Path

TARGET = Path("dashboard") / "app.py"


def main() -> None:

    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found. Run this from the project root.")
        return

    content = TARGET.read_text(encoding="utf-8")
    original_content = content

    # ------------------------------------------------------------------
    # Edit 1: status LOCKED -> ACTIVE for Phase 2
    # ------------------------------------------------------------------

    old_status = (
        '        "status": "LOCKED",\n'
        '        "items": [\n'
        '            "Parameter sweeps",\n'
        '            "Walk-forward",\n'
        '            "Monte Carlo",\n'
        '            "Overfit detection",\n'
        '        ],'
    )

    new_status = (
        '        "status": "ACTIVE",\n'
        '        "items": [\n'
        '            "Parameter sweeps",\n'
        '            "Walk-forward",\n'
        '            "Monte Carlo",\n'
        '            "Overfit detection",\n'
        '        ],'
    )

    if old_status not in content:
        print("STATUS BLOCK NOT FOUND -- no change made for edit 1.")
    else:
        content = content.replace(old_status, new_status, 1)
        print("Edit 1 (status LOCKED -> ACTIVE): OK")

    # ------------------------------------------------------------------
    # Edit 2: disabled button -> working page_link
    # ------------------------------------------------------------------

    old_button = (
        '    st.button(\n'
        '        "\U0001f512 Phase 2 Locked",\n'
        '        disabled=True,\n'
        '        use_container_width=True,\n'
        '    )'
    )

    new_button = (
        '    try:\n'
        '\n'
        '        st.page_link(\n'
        '            "pages/2_\u25c6_Phase2_Lab.py",\n'
        '            label="\u25c6 Open Phase 2 Lab",\n'
        '            icon="\U0001f9ec",\n'
        '            use_container_width=True,\n'
        '        )\n'
        '\n'
        '    except AttributeError:\n'
        '\n'
        '        st.info(\n'
        '            "\u25c6 Open **Phase 2 Lab** from the sidebar. "\n'
        '            "Upgrade Streamlit to 1.31+ to enable "\n'
        '            "the direct navigation button."\n'
        '        )'
    )

    if old_button not in content:
        print("BUTTON BLOCK NOT FOUND -- no change made for edit 2.")
    else:
        content = content.replace(old_button, new_button, 1)
        print("Edit 2 (locked button -> page_link): OK")

    # ------------------------------------------------------------------
    # Write back only if something changed
    # ------------------------------------------------------------------

    if content == original_content:
        print("No changes were made to the file.")
        return

    TARGET.write_text(content, encoding="utf-8")
    print(f"\nSaved changes to {TARGET}")


if __name__ == "__main__":
    main()