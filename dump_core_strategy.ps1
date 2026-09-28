$files = @(
    ".\strategy\swings.py",
    ".\strategy\structure.py",
    ".\strategy\liquidity.py",
    ".\strategy\zones.py",
    ".\strategy\signals.py",
    ".\strategy\retest_engine.py"
)
foreach ($f in $files) {
    Write-Output ""
    Write-Output "================================================================"
    Write-Output "FILE: $f"
    Write-Output "================================================================"
    Get-Content $f -Raw
}
