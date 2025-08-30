Param(
  [Parameter(Mandatory = $true)]
  [string]$Config
)

python train_maid.py --config $Config

if (Test-Path scripts/summarize_metrics.py) {
  python scripts/summarize_metrics.py (Join-Path (Split-Path $Config -Parent) 'metrics.json')
}

Write-Host "Done: " (Split-Path $Config -Parent)
