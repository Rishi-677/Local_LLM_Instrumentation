$ErrorActionPreference = "Stop"
$scriptDir = Split-Path $MyInvocation.MyCommand.Path
Set-Location (Join-Path $scriptDir "..")
$bin = "build/local_llm_instrumentation.exe"
if (-not (Test-Path $bin)) {
    Write-Error "Build not found"
    exit 1
}
$ver = "0.1.0"
$dist = "dist/Local_LLM_Instrumentation-$ver"
if (Test-Path $dist) { Remove-Item -Recurse -Force $dist }
New-Item -ItemType Directory -Path $dist -Force | Out-Null
Copy-Item $bin "$dist/"
$py_dir = "$dist/py_sidecar"
New-Item -ItemType Directory -Path $py_dir -Force | Out-Null
Get-ChildItem py_sidecar/*.py | ForEach-Object { Copy-Item $_ $py_dir/ }
Copy-Item local_llm_instrumentation.toml "$dist/"
New-Item -ItemType Directory -Path "$dist/docs" -Force | Out-Null
Copy-Item docs/implementation-roadmap.md "$dist/docs/"
$zip = "$dist.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path $dist -DestinationPath $zip
Write-Host "Created $zip"