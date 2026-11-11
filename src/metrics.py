"""Evaluation metrics for Stage 1 probability/ranking quality and quantum circuit diagnostics."""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int = 10) -> float:
    """Compute Precision@k after ranking by probability descending."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    top_idx = np.argsort(-y_prob, kind="mergesort")[:k]
    return float(np.mean(y_true[top_idx]))


def recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int = 10) -> float:
    """Compute Recall@k after ranking by probability descending."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    total_pos = np.sum(y_true)
    if total_pos == 0:
        return 0.0
    top_idx = np.argsort(-y_prob, kind="mergesort")[:k]
    return float(np.sum(y_true[top_idx]) / total_pos)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5
) -> float:
    """Compute quantile-binned Expected Calibration Error (ECE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if np.any(mask):
            ece += (np.sum(mask) / n) * abs(np.mean(y_true[mask]) - np.mean(y_prob[mask]))
    return float(ece)


def compute_stage1_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, k: int = 10
) -> Dict[str, float]:
    """Compute all required Stage 1 ranking and probability quality metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1.0 - 1e-7)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        f"precision_at_{k}": precision_at_k(y_true, y_prob, k=k),
        f"recall_at_{k}": recall_at_k(y_true, y_prob, k=k),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "ece_5bin": expected_calibration_error(y_true, y_prob, n_bins=5),
    }


def _gate_counts(circuit: Any) -> Dict[str, int]:
    """Count 1-qubit gates, 2-qubit gates, and SWAP gates excluding non-unitary ops."""
    one_q = 0
    two_q = 0
    swaps = 0
    ignored = {"measure", "barrier", "reset", "delay"}
    for inst in circuit.data:
        name = inst.operation.name.lower()
        if name in ignored:
            continue
        n_q = inst.operation.num_qubits
        if name == "swap":
            swaps += 1
        if n_q == 1:
            one_q += 1
        elif n_q == 2:
            two_q += 1
    return {
        "one_qubit_gates": int(one_q),
        "two_qubit_gates": int(two_q),
        "swap_gates": int(swaps),
    }


def extract_circuit_metrics(
    logical_circuit: Any,
    transpiled_circuit: Optional[Any] = None,
    active_physical_qubits: Optional[int] = None,
) -> Dict[str, Any]:
    """Extract logical/physical qubit counts, depths, gate counts, and SWAP counts."""
    logical_counts = _gate_counts(logical_circuit)
    metrics: Dict[str, Any] = {
        "logical_qubits": int(logical_circuit.num_qubits),
        "original_depth": int(logical_circuit.depth()),
        "original_one_qubit_gates": logical_counts["one_qubit_gates"],
        "original_two_qubit_gates": logical_counts["two_qubit_gates"],
        "original_swap_gates": logical_counts["swap_gates"],
        "original_ops": {str(k): int(v) for k, v in logical_circuit.count_ops().items()},
    }
    if transpiled_circuit is not None:
        trans_counts = _gate_counts(transpiled_circuit)
        # Count active physical qubits touched in transpiled circuit if backend layout expands width
        used_qubits = set()
        for inst in transpiled_circuit.data:
            if inst.operation.name.lower() != "barrier":
                for q in inst.qubits:
                    used_qubits.add(transpiled_circuit.find_bit(q).index)
        metrics.update(
            {
                "physical_qubits_backend": int(transpiled_circuit.num_qubits),
                "physical_qubits_active": int(
                    active_physical_qubits if active_physical_qubits is not None else len(used_qubits)
                ),
                "transpiled_depth": int(transpiled_circuit.depth()),
                "transpiled_one_qubit_gates": trans_counts["one_qubit_gates"],
                "transpiled_two_qubit_gates": trans_counts["two_qubit_gates"],
                "transpiled_swap_gates": trans_counts["swap_gates"],
                "transpiled_ops": {str(k): int(v) for k, v in transpiled_circuit.count_ops().items()},
            }
        )
    return metrics

