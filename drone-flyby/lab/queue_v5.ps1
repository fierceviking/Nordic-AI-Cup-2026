# Queue: v5 YOLO two-stage, then RF-DETR on the identical data.
#
# NOTE ON $ErrorActionPreference: it is deliberately NOT 'Stop'. Ultralytics
# writes its progress bars to stderr, and under 'Stop' PowerShell treats the
# first native stderr write as a terminating error, tears down the pipeline and
# kills the training child. That is exactly how the previous queue died at
# epoch 13 with no traceback. Failures are caught by $LASTEXITCODE and by
# checking for the artefact each step is supposed to produce.

$ErrorActionPreference = 'Continue'
$py = 'C:\Users\marti\miniconda3\envs\dm\python.exe'
$root = 'c:\Users\marti\Desktop\mtp_drone\drone-flyby'
Set-Location $root

function Step($number, $text) {
    Write-Host ''
    Write-Host ('=' * 68)
    Write-Host "[$number] $text    ($(Get-Date -Format 'HH:mm:ss'))"
    Write-Host ('=' * 68)
}

function Need($path, $what) {
    if (-not (Test-Path $path)) {
        Write-Host "QUEUE ABORTED: $what missing ($path)"
        exit 1
    }
}

Need 'lab\dataset_v5\data.yaml' 'dataset_v5'

# --- 1. YOLO v5 stage 1, resuming the run the last queue killed ------------ #
if (Test-Path 'lab\runs\v5s1\weights\last.pt') {
    Step 1 'YOLO v5 stage 1 (resuming)'
    & $py lab\train_yolo.py --model lab\runs\v5s1\weights\last.pt `
        --data lab\dataset_v5\data.yaml --name v5s1 --stage 1 --epochs 45 `
        --imgsz 960 --batch 12 --workers 6 --resume --monitor-every 0 `
        2>&1 | Out-File -Append lab\out\train_v5s1.log
} else {
    Step 1 'YOLO v5 stage 1 (fresh)'
    & $py lab\train_yolo.py --model yolo11s.pt `
        --data lab\dataset_v5\data.yaml --name v5s1 --stage 1 --epochs 45 `
        --imgsz 960 --batch 12 --workers 6 --monitor-every 0 `
        2>&1 | Out-File -Append lab\out\train_v5s1.log
}
Need 'lab\runs\v5s1\weights\best.pt' 'v5 stage 1 weights'

# --- 2. YOLO v5 stage 2 ---------------------------------------------------- #
Step 2 'YOLO v5 stage 2'
& $py lab\train_yolo.py --model lab\runs\v5s1\weights\best.pt `
    --data lab\dataset_v5\data_positives.yaml --name v5s2 --stage 2 `
    --epochs 15 --imgsz 960 --batch 12 --workers 6 --monitor-every 0 `
    2>&1 | Out-File -Append lab\out\train_v5s2.log
Need 'lab\runs\v5s2\weights\best.pt' 'v5 stage 2 weights'
Copy-Item lab\runs\v5s2\weights\best.pt model\v5s2.pt -Force
Write-Host 'v5 stage 2 done -> model\v5s2.pt'

# --- 3. RF-DETR on the same data ------------------------------------------- #
Step 3 'installing rfdetr (torch pinned)'
& $py -m pip install -c lab\constraints.txt rfdetr 2>&1 |
    Out-File -Append lab\out\rfdetr_install.log
& $py -c "import torch; assert torch.cuda.is_available(); print('torch OK', torch.__version__)"
if ($LASTEXITCODE -ne 0) {
    Write-Host 'QUEUE ABORTED: torch broken after rfdetr install - repair before continuing.'
    exit 1
}

Step 4 'converting dataset_v5 to COCO'
& $py lab\yolo_to_coco.py --root lab\dataset_v5 --out lab\dataset_v5_coco
Need 'lab\dataset_v5_coco\train\_annotations.coco.json' 'COCO conversion'

Step 5 'RF-DETR small @952 on dataset_v5'
& $py lab\train_rfdetr.py --data lab\dataset_v5_coco --size small `
    --resolution 952 --epochs 30 --batch 2 --accum 8 --name v5rfdetr `
    2>&1 | Out-File -Append lab\out\train_v5rfdetr.log

Step 6 'queue complete'
Write-Host 'YOLO   : model\v5s2.pt'
Write-Host 'RF-DETR: lab\runs\v5rfdetr\checkpoint_best_total.pth'
