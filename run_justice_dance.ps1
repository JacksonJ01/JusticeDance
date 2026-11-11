$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$downloads = Join-Path $env:USERPROFILE "Downloads"
$aihumoCandidates = @(
  (Join-Path $downloads "AIHuMo_refactored_v0_2"),
  (Join-Path $downloads "Humo_app")
)
$aihumoRoot = $aihumoCandidates | Where-Object { Test-Path (Join-Path $_ ".venv\Scripts\python.exe") } | Select-Object -First 1
if (-not $aihumoRoot) { throw "Could not find the AI Humo Python environment under your Downloads folder." }
& (Join-Path $aihumoRoot ".venv\Scripts\python.exe") (Join-Path $projectRoot "app\justice_dance.py")
