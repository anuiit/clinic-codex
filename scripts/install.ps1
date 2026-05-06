<#
.SYNOPSIS
Clinic Codex Windows installer - staged, CPU-only.
Each stage is checkpointed. If interrupted, re-run is safe (idempotent).
#>
param()

$ErrorActionPreference = 'Stop'

$ScriptPath = $MyInvocation.MyCommand.Path
$ScriptDir  = Split-Path -Parent $ScriptPath
$RepoRoot   = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

function Log([string]$msg)  { Write-Host "`n[install] $msg" }
function Fail([string]$msg) { Write-Host "[install] ERROR: $msg" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------------------
# Stage 0: Pre-flight checks
# ---------------------------------------------------------------------------
Log 'Stage 0/5: pre-flight checks'

$Python = $null
foreach ($candidate in @('py -3.11', 'py -3.10', 'python')) {
    $parts = $candidate -split ' '
    $cmd   = $parts[0]
    $args_ = if ($parts.Length -gt 1) { $parts[1..($parts.Length - 1)] } else { @() }
    try {
        $ver = & $cmd @args_ --version 2>&1
        if ($ver -match 'Python (3\.(10|11))') {
            $Python = @{ cmd = $cmd; args = $args_ }
            Log "Using $candidate ($ver)"
            break
        }
    } catch {}
}
if (-not $Python) { Fail "Need Python 3.10 or 3.11. Install from python.org and ensure the 'py' launcher is available." }

$verCheck = & $Python.cmd @($Python.args) -c 'import sys; print(str(sys.version_info.major) + "." + str(sys.version_info.minor))' 2>&1
if ($verCheck -notmatch '^3\.(10|11)$') { Fail "Python reports version $verCheck - need 3.10 or 3.11." }

$drive  = (Get-Item .).PSDrive.Name
$freeGB = [math]::Round((Get-PSDrive $drive).Free / 1GB, 1)
if ($freeGB -lt 2) { Fail "Need >=2GB free disk. Have ${freeGB}GB." }

# ---------------------------------------------------------------------------
# Stage 1: venv
# ---------------------------------------------------------------------------
Log 'Stage 1/5: venv at backend\.venv'

$VenvPy  = Join-Path $RepoRoot 'backend\.venv\Scripts\python.exe'
$VenvPip = Join-Path $RepoRoot 'backend\.venv\Scripts\pip.exe'

if (-not (Test-Path $VenvPy)) {
    & $Python.cmd @($Python.args) -m venv 'backend\.venv'
    if ($LASTEXITCODE -ne 0) { Fail 'venv creation failed' }
} else {
    Log '  reusing existing venv'
}

try { & $VenvPip install --quiet --upgrade pip 2>$null } catch { Log '  pip upgrade skipped' }

# ---------------------------------------------------------------------------
# Stage 2: core utils
# ---------------------------------------------------------------------------
Log 'Stage 2/5: core utils (numpy, pillow, scipy, pyyaml, tqdm)'

$pkgsCore = @(
    'numpy>=1.24,<2.5',
    'pillow>=10.0',
    'pyyaml>=6.0',
    'scipy>=1.11',
    'tqdm>=4.66'
)
& $VenvPip install --no-cache-dir --prefer-binary @pkgsCore
if ($LASTEXITCODE -ne 0) { Fail 'Stage 2 failed - core utils' }

# ---------------------------------------------------------------------------
# Stage 3: web (flask)
# ---------------------------------------------------------------------------
Log 'Stage 3/5: flask + flask-cors'

$pkgsWeb = @(
    'flask>=3.0,<4.0',
    'flask-cors>=4.0'
)
& $VenvPip install --no-cache-dir --prefer-binary @pkgsWeb
if ($LASTEXITCODE -ne 0) { Fail 'Stage 3 failed - flask' }

# ---------------------------------------------------------------------------
# Stage 4: torch CPU (~200MB - be patient)
# ---------------------------------------------------------------------------
Log 'Stage 4/5: torch CPU wheels (~200MB download - be patient)'

$pkgsTorch = @(
    'torch>=2.1,<2.6',
    'torchvision>=0.16,<0.21'
)
& $VenvPip install --no-cache-dir --prefer-binary --extra-index-url 'https://download.pytorch.org/whl/cpu' @pkgsTorch
if ($LASTEXITCODE -ne 0) { Fail 'Stage 4 failed - torch. Re-run script to resume.' }

$cudaVer = & $VenvPy -c 'import torch; print(torch.version.cuda)' 2>&1
if ($cudaVer -ne 'None') { Fail "torch installed with CUDA ($cudaVer) - should be CPU. Delete backend\.venv and retry." }

# ---------------------------------------------------------------------------
# Stage 5: SAM family + augmentation + timm
# ---------------------------------------------------------------------------
Log 'Stage 5/5: segment-anything, mobile-sam, albumentations, timm'

$pkgsSam = @(
    'segment-anything==1.0',
    'git+https://github.com/ChaoningZhang/MobileSAM.git',
    'albumentations>=1.4,<2.0',
    'timm>=0.9'
)
& $VenvPip install --no-cache-dir @pkgsSam
if ($LASTEXITCODE -ne 0) { Fail 'Stage 5 failed - SAM/augmentation' }

# ---------------------------------------------------------------------------
# Frontend (idempotent npm)
# ---------------------------------------------------------------------------
Log 'Frontend: npm install (idempotent)'

$nodeModules = Join-Path $RepoRoot 'frontend\node_modules'
$isEmpty     = (-not (Test-Path $nodeModules)) -or ((Get-ChildItem $nodeModules -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0)

if ($isEmpty) {
    Push-Location (Join-Path $RepoRoot 'frontend')
    npm install
    if ($LASTEXITCODE -ne 0) { Pop-Location; Fail 'npm install failed' }
    Pop-Location
} else {
    Log '  frontend\node_modules present - skipping (delete it to force reinstall)'
}

# ---------------------------------------------------------------------------
# Final import sanity
# ---------------------------------------------------------------------------
Log 'Sanity: importing all critical modules'

& $VenvPy -c 'import flask, torch, torchvision, mobile_sam, segment_anything, albumentations, timm; print("all imports OK")'
if ($LASTEXITCODE -ne 0) { Fail 'Sanity import failed - see error above' }

Log 'DONE. Launch with: scripts\run-dev.ps1 (or start backend and frontend manually)'
