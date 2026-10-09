param([switch]$GPU)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        python -m venv --without-pip .venv
        if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
    }
    python -m pip --python .venv\Scripts\python.exe install pip
    if ($LASTEXITCODE -ne 0) { throw 'pip installation failed' }
    if ($GPU) {
        .\.venv\Scripts\python.exe -m pip install --upgrade 'torch==2.14.1+cu130' --index-url https://download.pytorch.org/whl/cu130
        if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed' }
    }
    .\.venv\Scripts\python.exe -m pip install -r requirements-cpu.lock
    if ($LASTEXITCODE -ne 0) { throw 'Pinned dependency installation failed' }
    .\.venv\Scripts\python.exe -m pip install --no-deps -e '.[dev]'
    if ($LASTEXITCODE -ne 0) { throw 'Project installation failed' }
    .\.venv\Scripts\python.exe -m mjsrl doctor
} finally {
    Pop-Location
}
