$ErrorActionPreference = "Continue"
Write-Host "--- python playwright ---"
& C:\Python314\python.exe -c "import playwright, sys; print('playwright', playwright.__version__)" 2>&1 | Select-Object -First 3
Write-Host "--- npx playwright ---"
& npx --yes playwright --version 2>&1 | Select-Object -First 3
Write-Host "--- playwright browsers dir ---"
$pwdir = Join-Path $env:LOCALAPPDATA "ms-playwright"
if (Test-Path $pwdir) { Get-ChildItem $pwdir | Select-Object -ExpandProperty Name } else { Write-Host "(none)" }
Write-Host "--- chrome/edge processes ---"
Get-Process chrome, msedge -ErrorAction SilentlyContinue | Group-Object ProcessName | ForEach-Object { Write-Host ($_.Name + ": " + $_.Count) }
Write-Host "--- browsers installed ---"
foreach ($p in @("C:\Program Files\Google\Chrome\Application\chrome.exe",
                 "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                 "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                 "C:\Program Files\Microsoft\Edge\Application\msedge.exe")) {
  if (Test-Path $p) { Write-Host "FOUND: $p" }
}
Write-Host "--- user dir sanity ---"
Write-Host ("USERPROFILE = " + $env:USERPROFILE)
if (Test-Path (Join-Path $env:USERPROFILE "Documents")) { Write-Host "Documents exists" }
