# Independent replicate of run_mechanics.ps1 on the SAME seeds.
# The sim is not deterministic, so this is a second draw from the same
# distribution: it tests whether the ablation deltas survive resampling.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

"=== REPLICATE 2 (same seeds 11-20, horizon 3000) ==="
"replicate 1 was: none 1141.5 | no_clamp 1137.1 | no_aging 1255.7 | free_movement 1396.2 | no_predators 1766.9 | no_sprint_gate 1117.0"
""

foreach ($m in @("none", "no_sprint_gate", "no_predators", "free_movement", "no_aging", "no_clamp")) {
    $out = & "..\.venv\Scripts\python.exe" .\oracle_probe.py --seeds 11 12 13 14 15 16 17 18 19 20 `
        --workers 7 --horizon 3000 --mechanic $m 2>&1
    $mean = ($out | Select-String -Pattern "^mean").ToString()
    $deaths = ($out | Select-String -Pattern "^deaths").ToString()
    "{0,-16} {1}" -f $m, $mean
    "{0,-16} {1}" -f "", $deaths
}
"REPLICATE 2 COMPLETE"
