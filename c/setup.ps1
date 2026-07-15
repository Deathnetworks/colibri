# colibrì — Windows setup script
# Compiles the engine and performs a self-test.
Write-Host "🐦 colibrì — setup (Windows 11)"

$ErrorActionPreference = "Stop"

$cc = if ($env:CC) { $env:CC -replace '^"|"$','' } else { "gcc" }
$make_cmd = if ($env:MAKE) { $env:MAKE -replace '^"|"$','' } else { "make" }

# 1) Dependencies
$gcc = Get-Command $cc -ErrorAction SilentlyContinue
if (-not $gcc) {
    Write-Error "Compiler ($cc) is missing. Install via scoop (scoop install mingw-winlibs) or MSYS2."
}
$make = Get-Command $make_cmd -ErrorAction SilentlyContinue
if (-not $make) {
    Write-Error "Make ($make_cmd) is missing. Install via scoop (scoop install make) or MSYS2."
}

$gccVer = (& $cc -dumpversion)
Write-Host "  gcc: $gccVer"

Write-Host -NoNewline "  OpenMP: "
$tempCode = "int main(){return 0;}"
Set-Content -Path "_omp.c" -Value $tempCode
$ompTest = (Start-Process -FilePath $cc -ArgumentList "-fopenmp", "_omp.c", "-o", "_omp.exe" -PassThru -Wait -NoNewWindow)
if ($ompTest.ExitCode -eq 0) {
    Write-Host "ok"
} else {
    Write-Error "libgomp is missing or broken."
}
if (Test-Path "_omp.c") { Remove-Item "_omp.c" }
if (Test-Path "_omp.exe") { Remove-Item "_omp.exe" }

# 2) Build
Write-Host "  building engine (ARCH=native)..."
& $make_cmd -s glm ARCH="native"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed."
}

# 3) Self-test
if ((Test-Path "glm_tiny") -and (Test-Path "ref_glm.json")) {
    $env:SNAP = ".\glm_tiny"
    $env:TF = "1"
    $r = (& .\glm.exe 64 16 16 2>$null | Select-String "[0-9]+/[0-9]+ positions")
    Write-Host "  engine self-test: $($r)  (expected 32/32)"
}

# 4) System info
$mem = Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum
$ramGB = [math]::Round($mem.Sum / 1GB)
Write-Host "  RAM: ${ramGB} GB   (more RAM = more cached experts = faster inference)"
Write-Host ""
Write-Host "ready. Next steps:"
Write-Host "  .\coli build           # already done"
Write-Host "  .\coli convert --model D:\path\on\NVMe\glm52_i4     # generate the int4 model (hours)"
Write-Host "  .\coli info  --model D:\path\on\NVMe\glm52_i4"
Write-Host "  .\coli chat  --model D:\path\on\NVMe\glm52_i4 --ram <GB>"
Write-Host ""
Write-Host "IMPORTANT: keep the model on fast storage (NVMe/NTFS), never on a network mount."

# Optional GPU backend compilation warning
Write-Host "To build the GPU/XPU backends (CUDA, SYCL, Vulkan), run .\build_backends.ps1"
