@echo off
setlocal

set "MODE=quick"
set "DO_INSTALL=0"
set "TEST_ONLY=0"
set "HAS_ARGS=0"

:parse_args
if "%~1"=="" goto after_args
set "HAS_ARGS=1"
if /i "%~1"=="--full" goto set_full
if /i "%~1"=="-f" goto set_full
if /i "%~1"=="--quick" goto set_quick
if /i "%~1"=="-q" goto set_quick
if /i "%~1"=="--install" goto set_install
if /i "%~1"=="-i" goto set_install
if /i "%~1"=="--test" goto set_test
if /i "%~1"=="-t" goto set_test
if /i "%~1"=="--help" goto show_help
if /i "%~1"=="-h" goto show_help
if /i "%~1"=="/?" goto show_help

echo [WARNING] Unrecognized argument: %~1
shift
goto parse_args

:set_full
set "MODE=full"
shift
goto parse_args

:set_quick
set "MODE=quick"
shift
goto parse_args

:set_install
set "DO_INSTALL=1"
shift
goto parse_args

:set_test
set "TEST_ONLY=1"
shift
goto parse_args
if /i "%~1"=="--help" goto show_help
if /i "%~1"=="-h" goto show_help
if /i "%~1"=="/?" goto show_help

echo [WARNING] Unrecognized argument: %~1
shift
goto parse_args

:show_help
echo ==============================================================================
echo Trotterisation of a Spin Chain - Project Runner
echo ==============================================================================
echo.
echo Usage: run.bat [options]
echo.
echo Options:
echo   (no arguments)    Run quick verification of all phases and tests (~30s)
echo   --full, -f        Run full exhaustive simulations, sweeps and figures (~10-12 min)
echo   --quick, -q       Run quick smoke test mode (default)
echo   --test, -t        Run the pytest test suite only (225 tests)
echo   --install, -i     Install Python dependencies from requirements.txt first
echo   --help, -h        Show this help message
echo.
exit /b 0

:after_args
echo ==============================================================================
echo       Trotterisation of a Spin Chain - Full Project Execution
echo ==============================================================================
echo Mode: %MODE%
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not found in PATH.
    echo Please install Python 3.11+ and ensure python.exe is in your PATH.
    goto on_error
)

echo [*] Python interpreter:
python --version
echo.

if "%DO_INSTALL%"=="1" (
    echo [*] Installing requirements from requirements.txt...
    python -m pip install -r requirements.txt
    if errorlevel 1 goto on_error
    echo [*] Dependencies installed successfully.
    echo.
)

if "%TEST_ONLY%"=="1" (
    echo [*] Running Pytest suite [225 tests across 9 test suites]...
    python -m pytest tests/ -v
    if errorlevel 1 goto on_error
    echo.
    echo [PASS] All tests passed!
    goto on_success
)

rem Configure parameters based on execution mode
if "%MODE%"=="full" (
    set "PHASE3_ARGS="
    set "AUDIT_ARGS="
    set "SDK_ARGS="
    set "PHASE5_ARGS="
) else (
    set "PHASE3_ARGS=--quick --outdir results/quick_phase3 --no-plots"
    set "AUDIT_ARGS=--skip-robustness --no-figures"
    set "SDK_ARGS=--quick --no-figures"
    set "PHASE5_ARGS=--quick"
)

echo ==============================================================================
echo [Step 1/6] Running Test Suite [225 tests across 9 suites]
echo ==============================================================================
python -m pytest tests/ -q
if errorlevel 1 goto on_error
echo [PASS] All 225 unit and regression tests passed.
echo.

echo ==============================================================================
echo [Step 2/6] Phase 1 - Gate-Level Circuit Resource Estimation
echo ==============================================================================
python circuit_cost.py
if errorlevel 1 goto on_error
echo.

echo ==============================================================================
echo [Step 3/6] Phase 2 - Trotter Accuracy vs Gate Budgets
echo ==============================================================================
python -m analysis.run_phase2
if errorlevel 1 goto on_error
echo.

echo ==============================================================================
echo [Step 4/6] Phase 3 - Density-Matrix Simulation and Noise Models
echo ==============================================================================
python -m noise.run_phase3 %PHASE3_ARGS%
if errorlevel 1 goto on_error
echo.

echo ==============================================================================
echo [Step 5/6] Phase 4 - Independent Audit and Qiskit SDK Transpilation
echo ==============================================================================
echo [*] Running Stage 1-8 independent audit...
python -m validation.run_audit %AUDIT_ARGS%
if errorlevel 1 goto on_error
echo [*] Running Stage 8-13 Qiskit SDK transpilation...
python -m quantum_sdk.run_phase4 %SDK_ARGS%
if errorlevel 1 goto on_error
echo.

echo ==============================================================================
echo [Step 6/6] Phase 5 - Trajectories, Thermal Relaxation and Hardware Mitigation
echo ==============================================================================
python -m quantum_sdk.run_phase5 %PHASE5_ARGS%
if errorlevel 1 goto on_error
echo.

:on_success
echo ==============================================================================
echo [SUCCESS] The entire quantum computing project executed cleanly!
echo ==============================================================================
echo.
echo Deliverables and reports are ready:
echo   [Phase 1] Gate counts, logical depths, and circuit decomposition reports:
echo             - results/circuit_resources.csv
echo             - results/circuit_reports.json
echo.
echo   [Phase 2] Trotter accuracy vs gate budget scaling laws:
echo             - results/phase2/resource_efficiency.csv
echo             - results/phase2/minimum_resources_for_accuracy.csv
echo             - results/phase2/phase2_report.md
echo.
echo   [Phase 3] Open-system density matrix noisy simulation [n=4..6]:
echo             - results/phase3/gate_noise_results.csv
echo             - results/phase3/global_noise_results.csv
echo             - results/phase3/optimal_r.csv
echo             - results/phase3/phase3_report.md
echo.
echo   [Phase 4] Independent validation audit and Qiskit SDK compilation:
echo             - results/phase4/audit/audit_report.md
echo             - results/phase4/sdk/circuit_equivalence.csv
echo             - results/phase4/hardware_aware/compiled_resource_comparison.csv
echo             - results/phase4/phase4_report.md
echo.
echo   [Phase 5] Trajectories [n=10], Lindblad relaxation, Mock Falcon ZNE and Sabre:
echo             - results/phase5/dense_grid_optima.csv
echo             - results/phase5/thermal_relaxation_study.csv
echo             - results/phase5/trajectory_scaling.csv
echo             - results/phase5/hardware_execution.csv
echo             - results/phase5/routing_comparison.csv
echo             - results/phase5/figures/ [fig13..fig17]
echo             - results/phase5/phase5_report.md
echo.
echo Execution complete.
if "%HAS_ARGS%"=="0" pause
exit /b 0

:on_error
echo.
echo ==============================================================================
echo [ERROR] Project execution failed! Review the error output above.
echo ==============================================================================
if "%HAS_ARGS%"=="0" pause
exit /b 1
