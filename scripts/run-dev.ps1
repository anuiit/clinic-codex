<#
.SYNOPSIS
Clinic Codex dev launcher — backend on :7117, frontend on :7118.
#>
param()

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

function Log([string]$msg)  { Write-Host "[run-dev] $msg" }
function Fail([string]$msg) { Write-Host "[run-dev] ERROR: $msg" -ForegroundColor Red; exit 1 }

# --- Pre-flight: venv ---
$VenvDir = Join-Path $RepoRoot 'backend' '.venv'
$VenvPy  = Join-Path $VenvDir 'Scripts' 'python.exe'
if (-not (Test-Path $VenvPy)) {
    Fail "backend/.venv missing. Run: powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1"
}
& $VenvPy -c "import flask, torch, mobile_sam" 2>$null
if ($LASTEXITCODE -ne 0) { Fail "venv incomplete. Run: powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1" }

# --- Pre-flight: frontend deps ---
$NodeModules = Join-Path $RepoRoot 'frontend' 'node_modules'
if (-not (Test-Path $NodeModules)) {
    Fail "frontend/node_modules missing. Run: powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1"
}

# --- Pre-flight: prototypes (auto-export if missing) ---
$ProtoDerived = Join-Path $RepoRoot 'backend' 'codex_model' 'weights' 'prototypes.pt'
$ProtoSource  = Join-Path $RepoRoot 'backend' 'prototypes' 'prototypes.pt'
if (-not (Test-Path $ProtoDerived)) {
    if (-not (Test-Path $ProtoSource)) {
        Fail "Model artefacts missing: backend\prototypes\prototypes.pt not found. See backend\README.md."
    }
    Log "prototypes.pt missing — running export_model"
    Push-Location (Join-Path $RepoRoot 'backend')
    & $VenvPy -m codex_pipeline.scripts.export_model
    $exportExit = $LASTEXITCODE
    Pop-Location
    if ($exportExit -ne 0) { Fail "export_model failed — see error above" }
    if (-not (Test-Path $ProtoDerived)) { Fail "export_model ran but prototypes.pt still missing" }
}

# --- Pre-flight: ports ---
$BackendPort  = if ($env:BACKEND_PORT)  { $env:BACKEND_PORT }  else { '7117' }
$FrontendPort = if ($env:FRONTEND_PORT) { $env:FRONTEND_PORT } else { '7118' }
foreach ($port in @($BackendPort, $FrontendPort)) {
    $inUse = $false
    try {
        $tcp = New-Object Net.Sockets.TcpClient
        $tcp.Connect('127.0.0.1', [int]$port)
        $tcp.Close()
        $inUse = $true
    } catch {}
    if ($inUse) { Fail "Port $port already in use. Stop other process or override BACKEND_PORT/FRONTEND_PORT env." }
}

# --- Launch backend ---
Log "starting backend on :$BackendPort"
$env:PORT = $BackendPort
$backendProc = Start-Process -FilePath $VenvPy `
    -ArgumentList "backend\examples\flask_api.py" `
    -PassThru -NoNewWindow
$env:PORT = $null

# --- Launch frontend ---
Log "starting frontend on :$FrontendPort"
$env:PORT = $FrontendPort
$frontendProc = Start-Process -FilePath 'npm' `
    -ArgumentList 'run', 'dev' `
    -WorkingDirectory (Join-Path $RepoRoot 'frontend') `
    -PassThru -NoNewWindow
$env:PORT = $null

# --- Wait for backend ready ---
$ready = $false
for ($i = 1; $i -le 30; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$BackendPort/classes" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($r.StatusCode -eq 200) {
            Log "backend ready: http://localhost:$BackendPort"
            $ready = $true
            break
        }
    } catch {}
    if ($i -eq 30) { Fail "backend did not start within 30s — see logs above" }
}

Write-Host ""
Write-Host "[run-dev] both services up."
Write-Host "[run-dev]   backend:  http://localhost:$BackendPort"
Write-Host "[run-dev]   frontend: http://localhost:$FrontendPort"
Write-Host "[run-dev] Ctrl-C to stop."
Write-Host ""

# --- Wait + Cleanup ---
try {
    Wait-Process -Id $backendProc.Id, $frontendProc.Id
} finally {
    Log "shutting down"
    Stop-Process -Id $backendProc.Id  -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $frontendProc.Id -Force -ErrorAction SilentlyContinue
}
