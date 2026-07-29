# PowerShell script to run the micro-batch prototyping
Write-Host "Starting DeiT-LT Local Micro-Batch Training..." -ForegroundColor Green
Write-Host "Batch Size: 8 | Grad Accumulation: 16 | Mixed Precision: ON" -ForegroundColor Cyan

python train.py

Write-Host "Training Script Completed." -ForegroundColor Green
