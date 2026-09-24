"""
wire_phase2_components.py

Safely wires phase2_dashboard/components.py into
phase2_dashboard/app.py.

Changes:
1. Adds the components import exactly once.
2. Replaces the existing Phase 2 sidebar navigation block with
   components.render_nav_buttons().
3. Adds locked Phase 3 / Phase 4 navigation entries.
4. Renames the local diagnostic renderer to render_diagnostic().
5. Delegates diagnostic-card rendering to components.render_diagnostic_card().
6. Updates diagnostic renderer call sites safely.

This script does NOT change:
- runner.py
- results.py
- diagnostic execution logic
- persistence logic
- diagnostic definitions
- diagnostic calculations

Safe to run more than once -- each edit checks whether it was
already applied before touching anything, and a one-time backup
of the original file is written before any change is saved.

Run from the project root:

    python wire_phase2_components.py
"""

from pathlib import Path

TARGET = Path("phase2_dashboard") / "app.py"

BANNER = """
================================================================================
  PHASE 2 COMPONENT WIRING
  phase2_dashboard/components.py -> phase2_dashboard/app.py
================================================================================
"""


def replace_once(content, old, new, label):
    """
    Replace one exact block once.

    Returns (new_content, outcome) where outcome is one of:
    "ok", "not_found", "aborted_ambiguous".
    """

    if old not in content:
        print(f"  [ ]  {label}: not found")
        return content, "not_found"

    count = content.count(old)

    if count > 1:
        print(f"  [!]  {label}: ABORTED ({count} matches found; expected exactly 1)")
        return content, "aborted_ambiguous"

    content = content.replace(old, new, 1)
    print(f"  [x]  {label}: OK")
    return content, "ok"


def main():

    print(BANNER)

    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found.")
        print("Run this script from the project root.")
        return

    try:
        content = TARGET.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        print(f"ERROR: could not read {TARGET} as UTF-8: {exc}")
        return

    original_content = content
    outcomes = {}

    # ========================================================================
    # EDIT 1: components import, exactly once
    # ========================================================================

    import_line = "from phase2_dashboard import components"

    if import_line in content:
        print("  [x]  Edit 1 (import components): already applied")
        outcomes["import"] = "already"
    else:
        old_import = "import streamlit as st\n"
        new_import = (
            "import streamlit as st\n\n"
            "from phase2_dashboard import components\n"
        )

        if old_import not in content:
            print("  [ ]  Edit 1 (import components): not found")
            outcomes["import"] = "not_found"
        else:
            content = content.replace(old_import, new_import, 1)
            print("  [x]  Edit 1 (import components): OK")
            outcomes["import"] = "ok"

    # ========================================================================
    # EDIT 2: sidebar navigation loop -> components.render_nav_buttons()
    #          + locked Phase 3 / Phase 4 entries
    # ========================================================================

    old_nav = (
        '    pages = ["Dashboard", "Robustness", "Stability", "Costs", "Overfitting", "History"]\n'
        '\n'
        '    for page in pages:\n'
        '        if st.button(page, key=f"nav_{page}", use_container_width=True):\n'
        '            st.session_state.phase2_active_page = page\n'
        '            st.rerun()'
    )

    new_nav = (
        '    pages = [\n'
        '        "Dashboard",\n'
        '        "Robustness",\n'
        '        "Stability",\n'
        '        "Costs",\n'
        '        "Overfitting",\n'
        '        "History",\n'
        '    ]\n'
        '\n'
        '    clicked_page = components.render_nav_buttons(\n'
        '        pages,\n'
        '        st.session_state.phase2_active_page,\n'
        '        key_prefix="nav",\n'
        '    )\n'
        '\n'
        '    if clicked_page:\n'
        '        st.session_state.phase2_active_page = clicked_page\n'
        '        st.rerun()\n'
        '\n'
        '    components.render_locked_nav_item(\n'
        '        "Phase 3 Lab",\n'
        '        key_prefix="nav_locked",\n'
        '    )\n'
        '\n'
        '    components.render_locked_nav_item(\n'
        '        "Phase 4 Lab",\n'
        '        key_prefix="nav_locked",\n'
        '    )'
    )

    if new_nav in content:
        print("  [x]  Edit 2 (sidebar navigation): already applied")
        outcomes["nav"] = "already"
    else:
        content, outcomes["nav"] = replace_once(
            content, old_nav, new_nav, "Edit 2 (sidebar navigation)"
        )

    # ========================================================================
    # EDIT 3: local diagnostic-card renderer -> delegate to components.py
    # ========================================================================

    old_renderer = (
        'def render_diagnostic_card(key: str) -> None:\n'
        '\n'
        '    meta = DIAGNOSTIC_META[key]\n'
        '    status = get_latest_status(key)\n'
        '\n'
        '    st.markdown(\n'
        '        f"""\n'
        '        <div class="diagnostic-card">\n'
        '            <div class="diagnostic-number">TEST {meta["number"]}</div>\n'
        '            <div class="diagnostic-title">{meta["title"]}</div>\n'
        '            <div class="diagnostic-description">{meta["description"]}</div>\n'
        '            <div class="diagnostic-status">{status_html(status)}</div>\n'
        '        </div>\n'
        '        """,\n'
        '        unsafe_allow_html=True,\n'
        '    )\n'
        '\n'
        '    if st.button(\n'
        '        f"Run {meta[\'title\']}",\n'
        '        key=f"run_{key}",\n'
        '        use_container_width=True,\n'
        '        disabled=st.session_state.running,\n'
        '    ):\n'
        '        run_one(key)\n'
        '        st.rerun()'
    )

    new_renderer = (
        'def render_diagnostic(key: str) -> None:\n'
        '    """Render one diagnostic card and handle its Run button.\n'
        '\n'
        '    components.py owns the visual rendering.\n'
        '    app.py continues to own diagnostic execution and reruns.\n'
        '    """\n'
        '\n'
        '    meta = DIAGNOSTIC_META[key]\n'
        '    status = get_latest_status(key)\n'
        '\n'
        '    clicked = components.render_diagnostic_card(\n'
        '        number=meta["number"],\n'
        '        title=meta["title"],\n'
        '        description=meta["description"],\n'
        '        status=status,\n'
        '        run_label="Run",\n'
        '        run_key=f"run_{key}",\n'
        '        running=st.session_state.running,\n'
        '    )\n'
        '\n'
        '    if clicked:\n'
        '        run_one(key)\n'
        '        st.rerun()'
    )

    if new_renderer in content:
        print("  [x]  Edit 3 (diagnostic renderer): already applied")
        outcomes["renderer"] = "already"
    else:
        content, outcomes["renderer"] = replace_once(
            content, old_renderer, new_renderer, "Edit 3 (diagnostic renderer)"
        )

    # ========================================================================
    # EDIT 4: update call sites -- only the old name, so this is inert
    #          once Edit 3 has run
    # ========================================================================

    old_call = "render_diagnostic_card(key)"
    new_call = "render_diagnostic(key)"

    count = content.count(old_call)

    if count:
        content = content.replace(old_call, new_call)
        print(f"  [x]  Edit 4 (call sites: render_diagnostic_card(key)): OK ({count} replaced)")
        outcomes["calls"] = "ok"
    else:
        print("  [x]  Edit 4 (call sites: render_diagnostic_card(key)): already applied")
        outcomes["calls"] = "already"

    old_parameter_call = 'render_diagnostic_card("parameter_sensitivity")'
    new_parameter_call = 'render_diagnostic("parameter_sensitivity")'

    if old_parameter_call in content:
        content = content.replace(old_parameter_call, new_parameter_call)
        print('  [x]  Edit 5 (call site: "parameter_sensitivity"): OK')
        outcomes["param_call"] = "ok"
    else:
        print('  [x]  Edit 5 (call site: "parameter_sensitivity"): already applied')
        outcomes["param_call"] = "already"

    # ========================================================================
    # WRITE + BACKUP
    # ========================================================================

    print()

    if content == original_content:
        print("No changes were needed -- wiring is already fully applied.")
        return

    if "aborted_ambiguous" in outcomes.values():
        print("Some edits were ABORTED due to ambiguous matches.")
        print("No file was written. Review the [!] lines above before retrying.")
        return

    backup = TARGET.with_suffix(".py.wire_backup")

    if not backup.exists():
        backup.write_text(original_content, encoding="utf-8")
        print(f"Backup created: {backup}")

    TARGET.write_text(content, encoding="utf-8")

    print(f"Saved changes to: {TARGET}")
    print()
    print("--------------------------------------------------------------------------------")
    print("  SUMMARY")
    print("--------------------------------------------------------------------------------")
    applied = [k for k, v in outcomes.items() if v == "ok"]
    already = [k for k, v in outcomes.items() if v == "already"]
    missing = [k for k, v in outcomes.items() if v == "not_found"]
    print("  Applied this run : " + (", ".join(applied) if applied else "(none)"))
    print("  Already applied  : " + (", ".join(already) if already else "(none)"))
    if missing:
        print("  NOT FOUND        : " + ", ".join(missing) + "  <-- review before launching")
    print("--------------------------------------------------------------------------------")
    print()
    print("Next steps:")
    print("  1) python -m py_compile phase2_dashboard/app.py")
    print("  2) streamlit run .\\dashboard\\app.py")


if __name__ == "__main__":
    main()