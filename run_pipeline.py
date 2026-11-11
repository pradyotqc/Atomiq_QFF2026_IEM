"""End-to-end runner for the Hybrid Quantum-Classical Term Deposit Campaign Pipeline."""

from __future__ import annotations

import json
import platform
import shutil
import socket
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict

from src.config import (
    CV_SPLIT_SEED,
    GLOBAL_SEED,
    MODEL_SEED,
    MONTE_CARLO_SEED,
    MONTE_CARLO_SHOTS,
    OPTIMIZER_SEED,
    RESULTS_DIR,
    SIMULATOR_SEED,
    STAGE1_PREDICTIONS_CSV,
    STAGE1_RESULT_CSV,
    STAGE2_BUDGET_C,
    STAGE2_CAPACITY_H,
    STAGE2_EXACT_K,
    STAGE2_QAOA_SHOTS,
    STAGE2_RESULT_CSV,
    STAGE2_SELECTION_CSV,
    STAGE3_MIN_SUBSCRIBERS,
    STAGE3_RESULT_CSV,
    STAGE3_SHOTS,
    STAGE3_SIMULATION_CSV,
    TRANSPILER_SEED,
)
from src.data_loader import load_campaign_data
from src.plots import generate_all_plots
from src.stage1_predict import run_stage1
from src.stage2_optimize import run_stage2
from src.stage3_simulate import run_stage3


def _pkg_versions() -> Dict[str, str]:
    pkgs = [
        "qiskit",
        "qiskit-aer",
        "qiskit-algorithms",
        "qiskit-optimization",
        "qiskit-machine-learning",
        "scikit-learn",
        "pandas",
        "numpy",
        "scipy",
        "openpyxl",
        "matplotlib",
    ]
    out: Dict[str, str] = {}
    for p in pkgs:
        try:
            out[p] = version(p)
        except Exception:
            out[p] = "unknown"
    return out


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _save_csv_with_alias(df: Any, primary_path: Path, alias_path: Path) -> None:
    """Write deliverable CSV identically to both the challenge-spec and Devpost alias paths."""
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(primary_path, index=False)
    shutil.copyfile(primary_path, alias_path)


def main() -> None:
    t0 = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading workbook ===")
    data = load_campaign_data()
    train_df = data["train"]
    test_df = data["test"]
    stage2_data_df = data["stage2_data"]

    print("=== Stage 1: Subscription Prediction (5-fold Stratified CV + Final Test) ===")
    t1 = time.perf_counter()
    stage1_res = run_stage1(train_df, test_df)
    stage1_elapsed = time.perf_counter() - t1
    _save_csv_with_alias(stage1_res["result_df"], STAGE1_RESULT_CSV, STAGE1_PREDICTIONS_CSV)
    print(
        f"Stage 1 completed in {stage1_elapsed:.2f}s | "
        f"saved -> {STAGE1_RESULT_CSV} & {STAGE1_PREDICTIONS_CSV}"
    )

    print("=== Stage 2: Constrained Customer Selection (QUBO + XY-QAOA + Exact Benchmark) ===")
    t2 = time.perf_counter()
    stage2_res = run_stage2(stage1_res["result_df"], stage2_data_df)
    stage2_elapsed = time.perf_counter() - t2
    _save_csv_with_alias(stage2_res["result_df"], STAGE2_RESULT_CSV, STAGE2_SELECTION_CSV)
    print(
        f"Stage 2 completed in {stage2_elapsed:.2f}s | "
        f"saved -> {STAGE2_RESULT_CSV} & {STAGE2_SELECTION_CSV}"
    )

    print("=== Stage 3: 6-Qubit Campaign Outcome Simulation ===")
    t3 = time.perf_counter()
    stage3_res = run_stage3(stage2_res["result_df"])
    stage3_elapsed = time.perf_counter() - t3
    _save_csv_with_alias(stage3_res["result_df"], STAGE3_RESULT_CSV, STAGE3_SIMULATION_CSV)
    print(
        f"Stage 3 completed in {stage3_elapsed:.2f}s | "
        f"saved -> {STAGE3_RESULT_CSV} & {STAGE3_SIMULATION_CSV}"
    )

    print("=== Generating Figures ===")
    figure_paths = generate_all_plots(stage1_res, stage2_res, stage3_res)

    # Save structured JSON artifacts in results/
    _save_json(
        {
            "primary_model_name": stage1_res["primary_model_name"],
            "cv_summary": stage1_res["cv_summary"],
            "aer_kernel_diagnostics": stage1_res["aer_kernel_diagnostics"],
            "vqc_circuit_metrics": stage1_res["vqc_circuit_metrics"],
        },
        RESULTS_DIR / "stage1_cv_metrics.json",
    )
    _save_json(
        {
            "primary_solver_name": stage2_res["primary_solver_name"],
            "primary_selection": stage2_res["primary_selection"],
            "optimality_gap": stage2_res["optimality_gap"],
            "exact_benchmark": stage2_res["exact_benchmark"],
            "qubo_solvers": stage2_res["qubo_solvers"],
            "xy_qaoa_10q_benchmark": stage2_res["xy_qaoa_10q_benchmark"],
        },
        RESULTS_DIR / "stage2_summary.json",
    )
    _save_json(
        {
            "selected_ids": stage3_res["selected_ids"],
            "bit_ordering_convention": stage3_res["bit_ordering_convention"],
            "qubit_mapping": stage3_res["qubit_mapping"],
            "shots": stage3_res["shots"],
            "exact_poisson_binomial": stage3_res["exact_poisson_binomial"],
            "monte_carlo_100k": stage3_res["monte_carlo_100k"],
            "monte_carlo_matched_shots": stage3_res["monte_carlo_matched_shots"],
            "quantum_results": stage3_res["quantum_results"],
            "errors": stage3_res["errors"],
            "circuit_metrics_aer": stage3_res["circuit_metrics_aer"],
            "circuit_metrics_linear_backend": stage3_res["circuit_metrics_linear_backend"],
            "circuit_qasm_text": stage3_res["circuit_qasm_text"],
        },
        RESULTS_DIR / "stage3_summary.json",
    )

    total_elapsed = time.perf_counter() - t0
    _save_json(
        {
            "hostname": socket.gethostname(),
            "python_version": platform.python_version(),
            "package_versions": _pkg_versions(),
            "seeds": {
                "global_seed": GLOBAL_SEED,
                "cv_split_seed": CV_SPLIT_SEED,
                "model_seed": MODEL_SEED,
                "optimizer_seed": OPTIMIZER_SEED,
                "simulator_seed": SIMULATOR_SEED,
                "transpiler_seed": TRANSPILER_SEED,
                "monte_carlo_seed": MONTE_CARLO_SEED,
            },
            "parameters": {
                "stage2_exact_k": STAGE2_EXACT_K,
                "stage2_budget_c": STAGE2_BUDGET_C,
                "stage2_capacity_h": STAGE2_CAPACITY_H,
                "stage2_qaoa_shots": STAGE2_QAOA_SHOTS,
                "stage3_min_subscribers": STAGE3_MIN_SUBSCRIBERS,
                "stage3_shots": STAGE3_SHOTS,
                "monte_carlo_shots": MONTE_CARLO_SHOTS,
            },
            "timings_seconds": {
                "stage1": stage1_elapsed,
                "stage2": stage2_elapsed,
                "stage3": stage3_elapsed,
                "total": total_elapsed,
            },
            "figures": figure_paths,
        },
        RESULTS_DIR / "pipeline_metadata.json",
    )
    print(f"=== Pipeline finished cleanly in {total_elapsed:.2f}s ===")


if __name__ == "__main__":
    main()

