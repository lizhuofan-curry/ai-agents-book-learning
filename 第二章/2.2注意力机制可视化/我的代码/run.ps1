$ErrorActionPreference = 'Stop'
$experimentDir = Split-Path -Parent $PSScriptRoot
$chapterDir = Split-Path -Parent $experimentDir
$projectDir = Split-Path -Parent $chapterDir
$uvExe = 'E:\Hermes\bin\uv.exe'
$experimentPython = Join-Path $projectDir '.venv\Scripts\python.exe'
$env:UV_CACHE_DIR = Join-Path $projectDir '.uv-cache'
$env:HF_HOME = Join-Path $projectDir '.uv-cache\huggingface'
$env:TEMP = Join-Path $projectDir '.uv-cache\tmp'
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = '1'
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
Push-Location -LiteralPath $projectDir
try {
    & $uvExe sync --locked --python $experimentPython
    if ($LASTEXITCODE -ne 0) { throw 'Shared environment sync failed.' }
}
finally { Pop-Location }
& $experimentPython (Join-Path $PSScriptRoot 'attention_demo.py')
if ($LASTEXITCODE -ne 0) { throw 'Experiment failed; please share the error output.' }
