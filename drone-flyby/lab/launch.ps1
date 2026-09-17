# Launch the long-running jobs detached from the chat terminal.
#
# Processes started inside a chat terminal die when that terminal is cleaned
# up - which is how an RF-DETR run and the API server were both lost at 18:32
# with no checkpoint written. Start-Process gives them their own lifetime.
#
#   .\lab\launch.ps1 server   [weights] [alternates]
#   .\lab\launch.ps1 rfdetr
#   .\lab\launch.ps1 stop

param(
    [Parameter(Mandatory = $true)][ValidateSet('server', 'rfdetr', 'stop')]$What,
    $Weights = 'model\v5s2.pt',
    $Alternates = '2'
)

$py = 'C:\Users\marti\miniconda3\envs\dm\python.exe'
$root = 'c:\Users\marti\Desktop\mtp_drone\drone-flyby'
Set-Location $root

function Running($pattern) {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like $pattern }
}

if ($What -eq 'stop') {
    foreach ($p in @('*api.py*', '*train_rfdetr*')) {
        Running $p | ForEach-Object {
            Write-Host "stopping PID $($_.ProcessId)"
            Stop-Process -Id $_.ProcessId -Force
        }
    }
    return
}

if ($What -eq 'server') {
    Running '*api.py*' | ForEach-Object {
        Write-Host "stopping existing server PID $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force
    }
    Start-Sleep -Seconds 2
    $env:DRONE_WEIGHTS = $Weights
    $env:DRONE_ALTERNATES = $Alternates
    $env:DRONE_RECORD_DIR = 'lab\recordings'
    Start-Process -FilePath $py -ArgumentList 'api.py' -WorkingDirectory $root `
        -WindowStyle Hidden `
        -RedirectStandardOutput lab\out\server.log `
        -RedirectStandardError lab\out\server.err
    Start-Sleep -Seconds 12
    try {
        $h = Invoke-RestMethod http://127.0.0.1:9053/health -TimeoutSec 10
        Write-Host "server up: $($h.weights)  alternates=$($h.alternate_classes)  loaded=$($h.solver_loaded)"
    } catch {
        Write-Host 'server did not answer /health - check lab\out\server.err'
    }
    return
}

if ($What -eq 'rfdetr') {
    Running '*train_rfdetr*' | ForEach-Object {
        Write-Host "stopping existing rfdetr PID $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force
    }
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $args = 'lab\train_rfdetr.py --data lab\dataset_v5_coco --size small ' +
            '--resolution 952 --epochs 30 --batch 2 --accum 8 --name v5rfdetr'
    Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $root `
        -WindowStyle Hidden `
        -RedirectStandardOutput lab\out\train_v5rfdetr.log `
        -RedirectStandardError lab\out\train_v5rfdetr.err
    Start-Sleep -Seconds 5
    $p = Running '*train_rfdetr*'
    if ($p) { Write-Host "rf-detr training started, PID $($p.ProcessId)" }
    else { Write-Host 'rf-detr did not start - check lab\out\train_v5rfdetr.err' }
    return
}
