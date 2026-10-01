# CougarMap installer (Windows). Right-click -> "Run with PowerShell".
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "== Installing CougarMap =="
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "-- installing uv (Python tool manager, from astral.sh)"
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
$wheel = Get-ChildItem "$here\cougarmap-*.whl" | Sort-Object Name | Select-Object -Last 1
Write-Host "-- installing $($wheel.Name)"
uv tool install --force --python 3.12 $wheel.FullName
uv tool update-shell | Out-Null
$bin = (uv tool dir --bin).Trim()
Write-Host "-- connecting CougarMap to your AI tools"
& "$bin\cougarmap.exe" setup
Write-Host ""
Write-Host "Done. Restart your AI app and ask: 'Find me the cougar hotspots near Missoula, MT'"
Write-Host "Or open a new PowerShell window and type:  cougarmap hotspots ""Missoula, MT"""
Read-Host "Press Enter to close"
