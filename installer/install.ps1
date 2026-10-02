# CougarMap installer (Windows). Right-click -> "Run with PowerShell", or in PowerShell:
#   powershell -c "irm https://raw.githubusercontent.com/bryantclark/cougarmap/main/installer/install.ps1 | iex"
# Running it again updates CougarMap.
$ErrorActionPreference = "Stop"
Write-Host "== Installing CougarMap =="
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "-- installing uv (Python tool manager, from astral.sh)"
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
Write-Host "-- installing CougarMap (from PyPI)"
uv tool install --force --upgrade --python 3.12 cougarmap
uv tool update-shell | Out-Null
$bin = (uv tool dir --bin).Trim()
Write-Host "-- connecting CougarMap to your AI tools"
& "$bin\cougarmap.exe" setup
Write-Host ""
Write-Host "Done. Restart your AI app and ask: 'Find me the cougar hotspots near Missoula, MT'"
Write-Host "Or open a new PowerShell window and type:  cougarmap hotspots ""Missoula, MT"""
if ($PSCommandPath) { Read-Host "Press Enter to close" }  # double-clicked: keep the window open
