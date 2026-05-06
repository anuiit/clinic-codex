<#
Windows PowerShell installer (best-effort).
# Note: Windows support is best-effort
#>
param()

function Command-Exists {
    param([string]$cmd)
    $null -ne (Get-Command $cmd -ErrorAction SilentlyContinue)
}

Write-Output "Checking system prerequisites..."

if (Command-Exists -cmd "python") {
    $py = python --version 2>&1
    if ($py -match 'Python ([0-9]+)\.([0-9]+)') {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
            Write-Error "ERROR: Python >= 3.10 is required. Found: $py"
            exit 1
        }
    }
} else {
    Write-Error "ERROR: python not found. Please install Python 3.10+"
    exit 1
}

if (Command-Exists -cmd "node") {
    $node = node --version
    if ($node -match 'v([0-9]+)') {
        $nodeMajor = [int]$matches[1]
        if ($nodeMajor -lt 18) {
            Write-Error "ERROR: Node >= 18 is required. Found: $node"
            exit 1
        }
    }
} else {
    Write-Error "ERROR: node not found. Please install Node.js v18+"
    exit 1
}

Write-Output "Installing backend Python dependencies..."
if (Test-Path "backend/requirements.txt") {
    python -m pip install --upgrade pip
    python -m pip install -r backend/requirements.txt
} else {
    Write-Warning "backend/requirements.txt not found — skipping pip install"
}

if (Test-Path "frontend") {
    Write-Output "Installing frontend npm dependencies..."
    Push-Location frontend
    if (Command-Exists -cmd "npm") {
        npm install
    } else {
        Write-Error "ERROR: npm not found. Install Node.js which includes npm."
        Pop-Location
        exit 1
    }
    Pop-Location
} else {
    Write-Warning "frontend/ directory not found — skipping npm install"
}

Write-Output "`n✓ Installation complete. Next steps:`n  - Run the dev server: bash scripts/run-dev.sh`
