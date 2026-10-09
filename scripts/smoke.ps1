param([int]$Steps = 50)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Push-Location -LiteralPath $projectRoot
try {
    .\.venv\Scripts\python.exe -m mjsrl smoke --steps $Steps
    if ($LASTEXITCODE -ne 0) { throw 'Training smoke failed' }
} finally {
    Pop-Location
}
