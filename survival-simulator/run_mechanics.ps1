# Waits for the MPC run to finish, then measures d(score)/d(mechanic).
# Each ablation relaxes ONE simulator constraint with the oracle held fixed,
# so any change is the simulator's sensitivity, not a new policy.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

while (-not (Select-String -Path mpc30.log -Pattern "^mean" -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 30
}
"=== MPC 30s lookahead (greedy baseline on these seeds = 1141.5, se 85) ==="
Select-String -Path mpc30.log -Pattern "^mean|^deaths" | ForEach-Object { $_.Line }
""

foreach ($m in @("none", "no_clamp", "no_aging", "free_movement", "no_predators", "no_sprint_gate")) {
    $out = & "..\.venv\Scripts\python.exe" .\oracle_probe.py --seeds 11 12 13 14 15 16 17 18 19 20 `
        --workers 7 --horizon 3000 --mechanic $m 2>&1
    $mean = ($out | Select-String -Pattern "^mean").ToString()
    $deaths = ($out | Select-String -Pattern "^deaths").ToString()
    "{0,-16} {1}" -f $m, $mean
    "{0,-16} {1}" -f "", $deaths
}
"ALL MECHANIC ABLATIONS COMPLETE"
