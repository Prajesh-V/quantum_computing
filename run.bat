@echo off
setlocal enabledelayedexpansion

echo ==============================================================================
echo Trotterisation of a Spin Chain - Quantum Simulation Runner
echo ==============================================================================

rem Check if python is in PATH
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not found in PATH. Please install Python 3.11+ and add it to PATH.
    exit /b 1
)

set MODE=%1
if "%MODE%"=="" set MODE=help

if /i "%MODE%"=="help" goto :help
if /i "%MODE%"=="test" goto :test
if /i "%MODE%"=="phase1" goto :phase1
if /i "%MODE%"=="phase2" goto :phase2
if /i "%MODE%"=="phase3" goto :phase3
if /i "%MODE%"=="phase4" goto :phase4
if /i "%MODE%"=="phase5" goto :phase5
if /i "%MODE%"=="quick" goto :quick
if /i "%MODE%"=="all" goto :all

echo [ERROR] Unknown option: %MODE%
goto :help

:test
echo.
echo [1/1] Running complete test suite (225 tests)...
python -m pytest tests/ -q
if %ERRORLEVEL% neq 0 (
    echo [FAIL] Test suite failed.
    exit /b %ERRORLEVEL%
)
echo [PASS] All tests passed successfully.
exit /b 0

:phase1
echo.
echo [*] Running Phase 1: Circuit Resource Estimation...
python circuit_cost.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
echo [OK] Phase 1 completed.
exit /b 0

:phase2
echo.
echo [*] Running Phase 2: Error vs Circuit Resources...
python -m analysis.run_phase2
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
echo [OK] Phase 2 completed.
exit /b 0

:phase3
echo.
echo [*] Running Phase 3: Density-Matrix Simulation & Optimal Steps...
python -m noise.run_phase3
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
echo [OK] Phase 3 completed.
exit /b 0

:phase4
echo.
echo [*] Running Phase 4: Independent Audit & SDK Transpilation...
python -m validation.run_audit
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python -m quantum_sdk.run_phase4
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
echo [OK] Phase 4 completed.
exit /b 0

:phase5
echo.
echo [*] Running Phase 5: Large-scale Trajectories, Continuous Noise & Hardware Mitigation...
python -m quantum_sdk.run_phase5
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
echo [OK] Phase 5 completed.
exit /b 0

:quick
echo.
echo [*] Running Quick Verification Suite...
echo.
echo [1/3] Running tests...
python -m pytest tests/ -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo.
echo [2/3] Running Phase 4 SDK (quick)...
python -m quantum_sdk.run_phase4 --quick --no-figures
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo.
echo [3/3] Running Phase 5 (quick)...
python -m quantum_sdk.run_phase5 --quick
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo.
echo [SUCCESS] Quick verification completed cleanly.
exit /b 0

:all
echo.
echo [*] Running Full Verification and Pipeline (Phases 1-5)...
echo.
echo [1/6] Running full test suite...
python -m pytest tests/ -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo.
echo [2/6] Running Phase 1...
python circuit_cost.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo.
echo [3/6] Running Phase 2...
python -m analysis.run_phase2
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo.
echo [4/6] Running Phase 3...
python -m noise.run_phase3
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo.
echo [5/6] Running Phase 4 Audit & SDK...
python -m validation.run_audit
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python -m quantum_sdk.run_phase4
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo.
echo [6/6] Running Phase 5...
python -m quantum_sdk.run_phase5
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo.
echo [SUCCESS] Complete pipeline execution succeeded.
exit /b 0

:help
echo.
echo Usage: run.bat [COMMAND]
echo.
echo Available Commands:
echo   test     Run the full test suite (pytest tests/ -q)
echo   quick    Run tests, quick Phase 4 SDK and quick Phase 5 (~40s)
echo   phase1   Run Phase 1: Gate-level circuit resource estimation
echo   phase2   Run Phase 2: Trotter accuracy vs gate budgets
echo   phase3   Run Phase 3: Noisy density-matrix simulation
echo   phase4   Run Phase 4: Independent audit and SDK transpilation
echo   phase5   Run Phase 5: Trajectories, thermal relaxation, mock hardware, ZNE
echo   all      Run full test suite and full pipeline across all phases
echo   help     Display this help menu
echo.
exit /b 0
