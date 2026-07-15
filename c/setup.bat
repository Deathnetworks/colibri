@echo off
echo 🐦 colibrì — setup (Windows 11)

rem Setup environment variables
if "%CC%"=="" (
    set CC_CMD=gcc
) else (
    rem Strip quotes
    set CC_CMD=%CC:"=%
)

if "%MAKE%"=="" (
    set MAKE_CMD=make
) else (
    rem Strip quotes
    set MAKE_CMD=%MAKE:"=%
)

rem 1) Dependencies
"%CC_CMD%" --version >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Compiler ^(%CC_CMD%^) is missing. Install via scoop ^(scoop install mingw-winlibs^) or MSYS2.
    goto :eof
)
"%MAKE_CMD%" --version >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Make ^(%MAKE_CMD%^) is missing. Install via scoop ^(scoop install make^) or MSYS2.
    goto :eof
)

for /f "tokens=*" %%v in ('"%CC_CMD%" -dumpversion') do set GCC_VER=%%v
echo   gcc: %GCC_VER%

echo | set /p="  OpenMP: "
echo int main(){return 0;} > _omp.c
"%CC_CMD%" -fopenmp _omp.c -o _omp.exe >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo ok
) else (
    echo libgomp is missing or broken.
    del _omp.c >nul 2>&1
    goto :eof
)
del _omp.c _omp.exe >nul 2>&1

rem 2) Build standard engine
echo   building engine (ARCH=native)...
"%MAKE_CMD%" -s glm ARCH=native
if %ERRORLEVEL% neq 0 (
    echo Build failed.
    goto :eof
)

rem 3) Self-test
if exist "glm_tiny\" if exist "ref_glm.json" (
    set SNAP=.\glm_tiny
    set TF=1
    for /f "tokens=*" %%r in ('.\glm.exe 64 16 16 2^>^&1 ^| findstr /r "[0-9][0-9]*/[0-9][0-9]* positions"') do set R_RES=%%r
    echo   engine self-test: %R_RES%  (expected 32/32)
)

rem 4) Setup oneAPI for SYCL kernel build
echo.
echo   Setting up Intel oneAPI for SYCL backend...
if exist "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" (
    call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 --force
) else (
    echo   oneAPI setvars.bat not found at default location. Please run it manually to compile SYCL backend.
)

echo.
echo ready. Next steps:
echo   .\coli build           # already done
echo   .\coli convert --model D:\path\on\NVMe\glm52_i4     # generate the int4 model (hours)
echo   .\coli info  --model D:\path\on\NVMe\glm52_i4
echo   .\coli chat  --model D:\path\on\NVMe\glm52_i4 --ram ^<GB^>
echo.
echo IMPORTANT: keep the model on fast storage (NVMe/NTFS), never on a network mount.
echo.
echo To build the XPU backends (CUDA, SYCL, Vulkan), run .\build_backends.ps1
