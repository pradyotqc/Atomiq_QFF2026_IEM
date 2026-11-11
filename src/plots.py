"""Publication-quality visualizations for Stage 1, Stage 2, Stage 3, and end-to-end architecture."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from .config import FIGURES_DIR


plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 240,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11,
    }
)


def plot_stage1_cv_comparison(stage1_res: Dict[str, Any], out_dir: Path = FIGURES_DIR) -> Path:
    """Plot 5-fold Stratified CV comparison across Classical and Quantum/Hybrid classifiers."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cv = stage1_res["cv_summary"]
    models = list(cv.keys())
    primary_name = stage1_res.get("primary_model_name", "Quantum_Kernel_LogReg")

    label_map = {
        "Classical_LogReg": "Classical\nLogReg",
        "Classical_RF": "Classical\nRF",
        "Classical_RBF_SVC": "Classical\nRBF SVC",
        "Quantum_Kernel_LogReg": "Q-Kernel\nLogReg*",
        "Quantum_Kernel_SVC": "Q-Kernel\nSVC",
        "Hybrid_VQC": "Hybrid\nVQC",
    }
    short_names = [label_map.get(m, m.replace("_", "\n")) for m in models]

    colors = []
    for m in models:
        if m == primary_name:
            colors.append("#059669")  # Emerald for selected primary model
        elif "Quantum" in m or "VQC" in m:
            colors.append("#2563eb")  # Royal blue for quantum/hybrid challengers
        else:
            colors.append("#64748b")  # Cool slate gray for classical baselines

    metrics = [
        ("pr_auc", "PR-AUC (Higher is Better — Random Baseline = 0.117)", "{:.3f}", None),
        ("roc_auc", "ROC-AUC (Higher is Better — Ranking Separation)", "{:.3f}", None),
        ("precision_at_10", "Precision@10 (Higher is Better — Top-10 Cohort Hits)", "{:.3f}", None),
        ("brier_score", "Brier Score (Lower is Better — Zoomed [0.095–0.105])", "{:.4f}", (0.095, 0.1052)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 7.6))
    for ax, (key, title, fmt, fixed_ylim) in zip(axes.flat, metrics):
        vals = [cv[m]["oof_metrics"][key] for m in models]
        bars = ax.bar(range(len(models)), vals, color=colors, edgecolor="#0f172a", linewidth=0.9, width=0.64)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(short_names, fontsize=10, fontweight="bold")
        ax.set_title(title, fontweight="bold", fontsize=12, pad=10)
        ax.grid(axis="y", alpha=0.28, linestyle="--")
        if fixed_ylim is not None:
            ax.set_ylim(*fixed_ylim)
            offset = (fixed_ylim[1] - fixed_ylim[0]) * 0.035
        else:
            ymax = max(vals)
            ax.set_ylim(0, ymax * 1.19)
            offset = ymax * 0.025

        for bar, v, m in zip(bars, vals, models):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + offset,
                fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=10.5,
                fontweight="bold" if m == primary_name else "semibold",
                color="#047857" if m == primary_name else "#0f172a",
            )

    legend_handles = [
        Patch(facecolor="#64748b", edgecolor="#0f172a", label="Classical Baselines"),
        Patch(facecolor="#059669", edgecolor="#0f172a", label="Selected Primary (*): Quantum_Kernel_LogReg"),
        Patch(facecolor="#2563eb", edgecolor="#0f172a", label="Other Quantum / Hybrid Models"),
    ]
    fig.suptitle(
        "Stage 1: 5-Fold Stratified Out-of-Fold CV Benchmark (Leakage-Free, Seed = 42)",
        fontsize=14,
        fontweight="bold",
        y=0.975,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=3,
        frameon=True,
        facecolor="#f8fafc",
        edgecolor="#cbd5e1",
        fontsize=10.5,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.895])
    out_path = out_dir / "stage1_cv_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_stage1_calibration_and_top10(
    stage1_res: Dict[str, Any], out_dir: Path = FIGURES_DIR
) -> Path:
    """Plot zoomed out-of-fold calibration curves and Top-15 Stage 1 test customer probabilities."""
    out_dir.mkdir(parents=True, exist_ok=True)
    y_train = np.asarray(stage1_res["y_train"])
    oof = stage1_res["oof_predictions"]
    cv = stage1_res["cv_summary"]
    res_df = stage1_res["result_df"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4))

    for name, color in [
        ("Quantum_Kernel_LogReg", "#059669"),
        ("Hybrid_VQC", "#7c3aed"),
        ("Classical_LogReg", "#475569"),
        ("Classical_RF", "#2563eb"),
    ]:
        if name in oof:
            prob_true, prob_pred = calibration_curve(
                y_train, np.asarray(oof[name]), n_bins=5, strategy="quantile"
            )
            ece = cv[name]["oof_metrics"]["ece_5bin"]
            ax1.plot(
                prob_pred,
                prob_true,
                marker="o",
                markersize=6.5,
                linewidth=2.3,
                label=f"{name} (ECE={ece:.4f})",
                color=color,
            )

    ax1.plot([0, 0.34], [0, 0.34], "k--", alpha=0.6, linewidth=1.6, label="Perfect calibration (y = x)")
    ax1.set_xlim(0.0, 0.33)
    ax1.set_ylim(0.0, 0.33)
    ax1.set_xlabel("Mean Predicted Probability (5-Quantile Bins)", fontsize=11)
    ax1.set_ylabel("Empirical Positive Rate", fontsize=11)
    ax1.set_title("Zoomed Out-of-Fold Probability Calibration", fontweight="bold", fontsize=12.5)
    ax1.legend(fontsize=9.5, loc="upper left", frameon=True, facecolor="#f8fafc")
    ax1.grid(alpha=0.28, linestyle="--")

    top15 = res_df.head(15).iloc[::-1]
    bar_colors = ["#059669" if r <= 3 else "#2563eb" if f == 1 else "#cbd5e1" for r, f in zip(top15["rank"], top15["top_10_flag"])]
    bars = ax2.barh(top15["customer_id"], top15["predicted_probability"], color=bar_colors, edgecolor="#0f172a", linewidth=0.5)
    ax2.set_xlabel("Predicted Subscription Probability", fontsize=11)
    ax2.set_title("Stage 1 Test Ranking (Top 10 Highlighted)", fontweight="bold", fontsize=12.5)
    ax2.grid(axis="x", alpha=0.28, linestyle="--")
    for bar, v in zip(bars, top15["predicted_probability"]):
        ax2.text(
            v + 0.006,
            bar.get_y() + bar.get_height() / 2.0,
            f"{v:.3f}",
            va="center",
            fontsize=9.5,
            fontweight="semibold",
        )
    ax2.set_xlim(0, max(top15["predicted_probability"]) * 1.18)

    plt.tight_layout()
    out_path = out_dir / "stage1_calibration_and_top10.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_stage2_optimization(stage2_res: Dict[str, Any], out_dir: Path = FIGURES_DIR) -> Path:
    """Plot Stage 2 QUBO CVaR convergence, noise/mitigation fidelity, and selected customer portfolio."""
    out_dir.mkdir(parents=True, exist_ok=True)
    res_df = stage2_res["result_df"]
    qubo = stage2_res["qubo_solvers"]
    xy_bench = stage2_res["xy_qaoa_10q_benchmark"]
    hw_vqe = xy_bench["hardware_efficient_10q_vqe"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # Panel 1: CVaR convergence for 20-qubit QUBO solvers (add Ising offset to match total QUBO energy)
    ising_offset = float(qubo["qubo_details"]["ising_offset"])
    qaoa_hist = [float(e) + ising_offset for e in qubo["qubo_cvar_qaoa"]["cvar_energy_history"]]
    vqe_hist = [float(e) + ising_offset for e in qubo["qubo_cvar_sampling_vqe"]["cvar_energy_history"]]
    exact_energy = -stage2_res["exact_benchmark"]["optimal"]["objective_value"]
    axes[0].plot(qaoa_hist, label="20q QUBO CVaR-QAOA (p=1)", color="#2563eb", linewidth=1.8)
    axes[0].plot(vqe_hist, label="20q QUBO CVaR-SamplingVQE", color="#7c3aed", linewidth=1.8)
    axes[0].axhline(exact_energy, color="#059669", linestyle="--", label=f"Exact Ground ({exact_energy:.2f})")
    axes[0].set_xlabel("Optimizer Evaluation")
    axes[0].set_ylabel("Total QUBO CVaR Energy (alpha=0.15)")
    axes[0].set_title("20-Qubit QUBO Optimizer Convergence", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25, linestyle="--")

    # Panel 2: Ideal vs Finite-Shot vs Noisy Unmitigated vs Readout Mitigated (10q HW-VQE)
    regimes = [
        ("Ideal SV", hw_vqe["ideal_statevector"]),
        ("Shots (4096)", hw_vqe["noiseless_finite_shot"]),
        ("Noisy Unmit.", hw_vqe["noisy_unmitigated"]),
        ("Noisy Mitigated", hw_vqe["noisy_readout_mitigated"]),
    ]
    x_pos = np.arange(len(regimes))
    feas_mass = [r[1]["feasible_probability_mass"] for r in regimes]
    opt_mass = [r[1]["exact_optimum_probability"] for r in regimes]
    w = 0.35
    axes[1].bar(x_pos - w / 2, feas_mass, width=w, label="Feasible Prob Mass", color="#0ea5e9")
    axes[1].bar(x_pos + w / 2, opt_mass, width=w, label="Exact Optimum Prob", color="#2563eb")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([r[0] for r in regimes], fontsize=8.5)
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("Probability")
    axes[1].set_title("Noise & Readout Mitigation (10q CVaR-VQE)", fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")

    # Panel 3: Top-10 expected net value with Selected 6 highlighted
    df_sorted = res_df.sort_values("expected_net_value", ascending=True)
    bar_cols = ["#059669" if s == 1 else "#cbd5e1" for s in df_sorted["selected_flag"]]
    bars = axes[2].barh(df_sorted["customer_id"], df_sorted["expected_net_value"], color=bar_cols)
    axes[2].set_xlabel("Expected Net Value (p * Deposit - Cost)")
    axes[2].set_title(
        f"Selected 6 of 10 (Obj={stage2_res['primary_selection']['objective_value']:.2f}, Gap={stage2_res['optimality_gap']:.3f})",
        fontweight="bold",
        fontsize=10,
    )
    axes[2].grid(axis="x", alpha=0.25, linestyle="--")
    for bar, (_, row) in zip(bars, df_sorted.iterrows()):
        axes[2].text(
            row["expected_net_value"] + 0.12,
            bar.get_y() + bar.get_height() / 2.0,
            f"c={int(row['contact_cost'])}, w={int(row['capacity_units'])}",
            va="center",
            fontsize=7.5,
        )
    axes[2].set_xlim(min(0, df_sorted["expected_net_value"].min() - 1), df_sorted["expected_net_value"].max() * 1.28)

    plt.tight_layout()
    out_path = out_dir / "stage2_optimization_landscape.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_stage3_outcomes(stage3_res: Dict[str, Any], out_dir: Path = FIGURES_DIR) -> Path:
    """Plot Stage 3 subscriber count PMF and P(K >= 3) comparison across classical/quantum regimes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    exact = stage3_res["exact_poisson_binomial"]
    mc = stage3_res["monte_carlo_100k"]
    qres = stage3_res["quantum_results"]

    ks = np.arange(7)
    pmf_exact = [exact["pmf_k_0_to_n"][str(k)] for k in ks]
    pmf_shots = [qres["noiseless_finite_shot"]["pmf"][str(k)] for k in ks]
    pmf_noisy = [qres["noisy_unmitigated"]["pmf"][str(k)] for k in ks]
    pmf_mit = [qres["noisy_readout_mitigated"]["pmf"][str(k)] for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    w = 0.2
    ax1.bar(ks - 1.5 * w, pmf_exact, width=w, label="Exact Poisson-Binomial", color="#0f172a")
    ax1.bar(ks - 0.5 * w, pmf_shots, width=w, label="Aer Ideal Shots (8192)", color="#2563eb")
    ax1.bar(ks + 0.5 * w, pmf_noisy, width=w, label="Aer Noisy Unmitigated", color="#dc2626")
    ax1.bar(ks + 1.5 * w, pmf_mit, width=w, label="Aer Noisy Mitigated", color="#059669")
    ax1.axvline(2.5, color="#f59e0b", linestyle="--", linewidth=1.8, label="Target Threshold (K >= 3)")
    ax1.set_xticks(ks)
    ax1.set_xlabel("Number of Subscribing Customers (out of 6)")
    ax1.set_ylabel("Probability Mass")
    ax1.set_title("Stage 3: Campaign Subscription Distribution (K = 0..6)", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.25, linestyle="--")

    labels = [
        "Exact\nPoisson-Binom",
        "Monte Carlo\n(100k)",
        "Quantum\nIdeal SV",
        "Quantum\nAer Shots",
        "Noisy\nUnmitigated",
        "Noisy\nMitigated",
    ]
    vals = [
        exact["exact_probability_at_least_k"],
        mc["probability_at_least_k"],
        qres["ideal_statevector"]["probability_at_least_3"],
        qres["noiseless_finite_shot"]["probability_at_least_3"],
        qres["noisy_unmitigated"]["probability_at_least_3"],
        qres["noisy_readout_mitigated"]["probability_at_least_3"],
    ]
    bar_cols = ["#0f172a", "#475569", "#7c3aed", "#2563eb", "#dc2626", "#059669"]
    bars = ax2.bar(range(len(labels)), vals, color=bar_cols, edgecolor="#1e293b", linewidth=0.7)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("P(At Least 3 Subscribers)")
    ax2.set_title("Stage 3: Success Probability Comparison", fontweight="bold")
    ax2.grid(axis="y", alpha=0.25, linestyle="--")
    for bar, v in zip(bars, vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + max(vals) * 0.02,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax2.set_ylim(0, max(vals) * 1.18)

    plt.tight_layout()
    out_path = out_dir / "stage3_circuit_and_outcomes.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_all_plots(
    stage1_res: Dict[str, Any],
    stage2_res: Dict[str, Any],
    stage3_res: Dict[str, Any],
    out_dir: Path = FIGURES_DIR,
) -> Dict[str, str]:
    """Generate all diagnostic and presentation plots."""
    p1 = plot_stage1_cv_comparison(stage1_res, out_dir=out_dir)
    p2 = plot_stage1_calibration_and_top10(stage1_res, out_dir=out_dir)
    p3 = plot_stage2_optimization(stage2_res, out_dir=out_dir)
    p4 = plot_stage3_outcomes(stage3_res, out_dir=out_dir)
    return {
        "stage1_cv_comparison": str(p1),
        "stage1_calibration_and_top10": str(p2),
        "stage2_optimization_landscape": str(p3),
        "stage3_circuit_and_outcomes": str(p4),
    }

