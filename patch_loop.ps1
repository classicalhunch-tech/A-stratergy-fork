$ErrorActionPreference = "Stop"
$path = ".\phase_04_live\runtime\loop.py"
$content = Get-Content -Raw -Path $path

$oldA = "if __name__ == `"__main__`":"
$newA = @"
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 4 live runtime loop.")
    parser.add_argument("--live", action="store_true", help="Submit real orders (dry_run=False). Requires manual verification of VERIFY items first.")
    parser.add_argument("--dry-run", action="store_true", help="Explicit no-op; dry-run is already the default.")
    args = parser.parse_args()

    if args.live and args.dry_run:
        raise SystemExit("Cannot pass both --live and --dry-run.")

"@
if ($content -notlike "*$oldA*") { Write-Error "Anchor A not found - aborting, nothing changed."; exit 1 }
$content = $content.Replace($oldA, $newA)

$oldB = "        dry_run=False,`r`n    )"
$newB = "        dry_run=not args.live,`r`n    )"
if ($content -notlike "*$oldB*") { Write-Error "Anchor B not found - aborting, check loop.py manually."; exit 1 }
$content = $content.Replace($oldB, $newB)

Set-Content -Path $path -Value $content -NoNewline
python -m py_compile $path
if ($LASTEXITCODE -ne 0) { Write-Error "Syntax check FAILED after patch - check loop.py manually."; exit 1 }
Write-Host "loop.py patched and syntax-verified successfully."
