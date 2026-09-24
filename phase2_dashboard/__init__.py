"""
phase2_dashboard/__init__.py

Phase 2 Robustness Lab dashboard package.

This package contains the Phase 2 dashboard interface and
its supporting runner/result components.

Responsibilities
----------------
- Provide the Phase 2 dashboard package.
- Connect the dashboard UI to existing Phase 2 diagnostics.
- Display and organize Phase 2 robustness results.

This package does NOT:
- modify strategy logic
- modify the canonical backtest
- modify Phase 1
- reimplement Phase 2 diagnostic calculations
- alter existing diagnostic caches

The existing modules in phase_02_optimization remain the
source of truth for all Phase 2 diagnostic calculations.
"""