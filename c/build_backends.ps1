param (
    [switch]$Sycl,
    [switch]$Vulkan
)

# SYCL Build
if ($Sycl) {
    Write-Host "Building SYCL Backend (coli_sycl.dll)..."
    # Execute the build in a subshell where we call the oneAPI setup script first
    $syclCommand = "cmd.exe /c `"call `""C:\Program Files (x86)\Intel\oneAPI\setvars.bat`"" intel64 --force && icpx -fsycl -O3 -shared -DCOLI_SYCL_BUILDING_DLL backend_sycl.cpp -o coli_sycl.dll`""
    Invoke-Expression $syclCommand
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to build SYCL backend."
    } else {
        Write-Host "Successfully built coli_sycl.dll"
    }
}

# Vulkan Build
if ($Vulkan) {
    Write-Host "Building Vulkan Backend (coli_vulkan.dll)..."
    # Assuming VULKAN_SDK is installed and cl is available
    $vulkanCommand = "cl.exe /O2 /LD /DCOLI_VULKAN_BUILDING_DLL backend_vulkan.cpp /link /OUT:coli_vulkan.dll /LIBPATH:\"`$env:VULKAN_SDK\lib\" vulkan-1.lib"
    Invoke-Expression $vulkanCommand
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to build Vulkan backend."
    } else {
        Write-Host "Successfully built coli_vulkan.dll"
    }
}

if (-not $Sycl -and -not $Vulkan) {
    Write-Host "Usage: .\build_backends.ps1 [-Sycl] [-Vulkan]"
}
