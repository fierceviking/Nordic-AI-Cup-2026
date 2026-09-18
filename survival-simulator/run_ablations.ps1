# Prices each piece of privileged oracle knowledge at n=10.
# Run:  powershell -ExecutionPolicy Bypass -File .\run_ablations.ps1
$ErrorActionPreference = "Continue"
$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$probe  = Join-Path $PSScriptRoot "oracle_probe.py"
$seeds  = @("11","12","13","14","15","16","17","18","19","20")

$conditions = [ordered]@{
    "full"            = @()
    "tree_age"        = @("--ablate", "tree_age")
    "predator_state"  = @("--ablate", "predator_state")
    "fruit_value"     = @("--ablate", "fruit_value")
}

foreach ($name in $conditions.Keys) {
    $started = Get-Date
    $argv = @($probe, "--seeds") + $seeds + @("--workers", "7", "--horizon", "3000") + $conditions[$name]
    $output = & $python $argv 2>&1
    $secs = [math]::Round(((Get-Date) - $started).TotalSeconds)
    Write-Output "=== $name  [${secs}s] ==="
    $output | Where-Object { $_ -match '^(mean|\{)' } | ForEach-Object { Write-Output $_ }
    Write-Output ""
}
Write-Output "ALL CONDITIONS COMPLETE"
