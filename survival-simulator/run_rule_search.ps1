$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Output "=== RULE SEARCH ==="
Write-Output "target: lower predator-induced lineage loss without losing foraging"
Write-Output "ablation hierarchy: predators +625 | movement +255 | aging +114 | gate -25 (null) | storage -4 (null)"
Write-Output "stage seeds 4 -> 12 -> 30 (common within stage, baseline in every batch)"
Write-Output "final check on unseen seeds 201-230"
Write-Output ""

# One line, no backtick continuations: PowerShell otherwise also runs the
# script through its file association, on the wrong interpreter.
& "..\.venv\Scripts\python.exe" .\rule_search.py --generations 4 --population 32 --horizon 3000 --workers 6 --out rule_search.json

Write-Output ""
Write-Output "RULE SEARCH COMPLETE"
