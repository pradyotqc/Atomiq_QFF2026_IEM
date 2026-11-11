"""Stage 3: 6-Qubit Campaign Outcome Quantum Circuit, Exact Poisson-Binomial, Monte Carlo & Noise Mitigation."""

from __future__ import annotations

import json
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

from .config import (
    MONTE_CARLO_SEED,
    MONTE_CARLO_SHOTS,
    SIMULATOR_SEED,
    STAGE3_MIN_SUBSCRIBERS,
    STAGE3_SHOTS,
    TRANSPILER_SEED,
)
from .metrics import extract_circuit_metrics


def _build_stage3_noise_model() -> NoiseModel:
    """Build realistic superconducting gate + asymmetric readout error model for 6-qubit simulation."""
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(0.001, 1), ["rx", "ry", "rz", "sx", "x", "h", "p", "u"]
    )
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(0.01, 2), ["cx", "ecr", "cz"]
    )
    readout = ReadoutError([[0.985, 0.015], [0.025, 0.975]])
    noise_model.add_all_qubit_readout_error(readout)
    return noise_model


def exact_poisson_binomial_at_least_k(
    probs: np.ndarray, min_k: int = STAGE3_MIN_SUBSCRIBERS
) -> Dict[str, Any]:
    """Compute exact P(sum Y_j >= min_k) and full PMF P(K = k) for independent Bernoullis."""
    n = len(probs)
    pmf = np.zeros(n + 1, dtype=float)
    pmf[0] = 1.0
    for p in probs:
        next_pmf = np.zeros(n + 1, dtype=float)
        next_pmf[:-1] += pmf[:-1] * (1.0 - p)
        next_pmf[1:] += pmf[:-1] * p
        pmf = next_pmf

    # Also cross-check over all 2^n states explicitly
    exact_sum_states = 0.0
    for idx in range(1 << n):
        hw = idx.bit_count()
        if hw >= min_k:
            prob_state = 1.0
            for j in range(n):
                bit = (idx >> j) & 1
                prob_state *= probs[j] if bit == 1 else (1.0 - probs[j])
            exact_sum_states += prob_state

    p_success = float(np.sum(pmf[min_k:]))
    expected_subscribers = float(np.sum(probs))
    return {
        "exact_probability_at_least_k": p_success,
        "exact_state_enum_check": float(exact_sum_states),
        "expected_subscribers": expected_subscribers,
        "pmf_k_0_to_n": {str(k): float(pmf[k]) for k in range(n + 1)},
    }


def monte_carlo_campaign_outcome(
    probs: np.ndarray,
    min_k: int = STAGE3_MIN_SUBSCRIBERS,
    n_trials: int = MONTE_CARLO_SHOTS,
    seed: int = MONTE_CARLO_SEED,
) -> Dict[str, Any]:
    """Simulate independent Bernoulli customer subscriptions via classical Monte Carlo."""
    rng = np.random.default_rng(seed)
    draws = rng.random((n_trials, len(probs))) < probs[None, :]
    counts_per_trial = np.sum(draws, axis=1)
    p_est = float(np.mean(counts_per_trial >= min_k))
    se = float(np.sqrt(p_est * (1.0 - p_est) / n_trials))
    pmf = {
        str(k): float(np.mean(counts_per_trial == k)) for k in range(len(probs) + 1)
    }
    return {
        "n_trials": int(n_trials),
        "seed": int(seed),
        "probability_at_least_k": p_est,
        "standard_error": se,
        "pmf_k_0_to_n": pmf,
    }


def build_stage3_circuit(
    probs: np.ndarray, add_measurements: bool = True
) -> tuple[QuantumCircuit, np.ndarray]:
    """Construct 6-qubit Ry(theta_j) state preparation circuit for Stage 3."""
    n = len(probs)
    thetas = 2.0 * np.arcsin(np.sqrt(np.clip(probs, 0.0, 1.0)))
    qc = QuantumCircuit(n, name="Stage3CampaignSim")
    for q in range(n):
        qc.ry(float(thetas[q]), q)
    if add_measurements:
        qc.measure_all()
    return qc, thetas


def _success_prob_from_counts(
    counts: Dict[str, int], n_qubits: int, shots: int, min_k: int
) -> tuple[float, Dict[str, float]]:
    """Decode Qiskit little-endian bitstrings ('q5..q0') and compute P(hw >= min_k)."""
    p_success = 0.0
    pmf = np.zeros(n_qubits + 1, dtype=float)
    for bstr, cnt in counts.items():
        # Verify bitstring length and count '1's explicitly per qubit index j -> bstr[n_qubits - 1 - j]
        clean = bstr.replace(" ", "")
        hw = sum(int(clean[n_qubits - 1 - j]) for j in range(n_qubits))
        prob = cnt / shots
        pmf[hw] += prob
        if hw >= min_k:
            p_success += prob
    return float(p_success), {str(k): float(pmf[k]) for k in range(n_qubits + 1)}


def _success_prob_from_dist(
    probs_vec: np.ndarray, n_qubits: int, min_k: int
) -> tuple[float, Dict[str, float]]:
    """Compute P(hw >= min_k) and PMF from length-2^n probability vector (little-endian index)."""
    p_success = 0.0
    pmf = np.zeros(n_qubits + 1, dtype=float)
    for idx, p in enumerate(probs_vec):
        hw = idx.bit_count()
        pmf[hw] += float(p)
        if hw >= min_k:
            p_success += float(p)
    return float(p_success), {str(k): float(pmf[k]) for k in range(n_qubits + 1)}


def run_stage3(
    stage2_result_df: pd.DataFrame,
    min_subscribers: int = STAGE3_MIN_SUBSCRIBERS,
    shots: int = STAGE3_SHOTS,
) -> Dict[str, Any]:
    """Run Stage 3 quantum simulation, exact Poisson-Binomial, Monte Carlo, and error mitigation."""
    selected_df = (
        stage2_result_df[stage2_result_df["selected_flag"] == 1]
        .sort_values(by=["rank", "customer_id"], ascending=[True, True])
        .reset_index(drop=True)
    )
    if len(selected_df) != 6:
        raise ValueError(f"Expected 6 selected customers for Stage 3, got {len(selected_df)}")

    ids = selected_df["customer_id"].astype(str).tolist()
    probs = selected_df["predicted_probability"].to_numpy(dtype=float)
    n_qubits = len(probs)

    # 1. Exact Classical Poisson-Binomial
    exact_res = exact_poisson_binomial_at_least_k(probs, min_k=min_subscribers)
    p_exact = exact_res["exact_probability_at_least_k"]

    # 2. Classical Monte Carlo (both 100k reference and matched-shot 8192 check)
    mc_100k = monte_carlo_campaign_outcome(
        probs, min_k=min_subscribers, n_trials=MONTE_CARLO_SHOTS, seed=MONTE_CARLO_SEED
    )
    mc_matched = monte_carlo_campaign_outcome(
        probs, min_k=min_subscribers, n_trials=shots, seed=MONTE_CARLO_SEED
    )

    # 3. Quantum Circuit Construction
    qc_nom, thetas = build_stage3_circuit(probs, add_measurements=False)
    qc_meas, _ = build_stage3_circuit(probs, add_measurements=True)

    # 3a. Ideal Statevector (exact mathematical quantum simulation)
    sv = Statevector.from_instruction(qc_nom)
    probs_sv = sv.probabilities()
    p_ideal_sv, pmf_ideal_sv = _success_prob_from_dist(probs_sv, n_qubits, min_subscribers)

    # 3b. Noiseless Finite-Shot Aer simulation
    aer_ideal = AerSimulator()
    pm_ideal = generate_preset_pass_manager(
        optimization_level=1, backend=aer_ideal, seed_transpiler=TRANSPILER_SEED
    )
    t_qc_ideal = pm_ideal.run(qc_meas)
    sampler_shots = AerSamplerV2(default_shots=shots, seed=SIMULATOR_SEED)
    res_shots = sampler_shots.run([t_qc_ideal]).result()[0]
    counts_shots = {str(k): int(v) for k, v in res_shots.data.meas.get_counts().items()}
    p_shots, pmf_shots = _success_prob_from_counts(counts_shots, n_qubits, shots, min_subscribers)

    # Also evaluate transpilation onto a realistic linear-coupled 6-qubit GenericBackendV2
    fake_backend = GenericBackendV2(num_qubits=n_qubits, coupling_map=[[i, i + 1] for i in range(n_qubits - 1)], seed=TRANSPILER_SEED)
    t_qc_hw = transpile(qc_meas, backend=fake_backend, optimization_level=1, seed_transpiler=TRANSPILER_SEED)
    circ_metrics_aer = extract_circuit_metrics(qc_meas, t_qc_ideal)
    circ_metrics_hw = extract_circuit_metrics(qc_meas, t_qc_hw)

    # 3c. Noisy Aer Simulation (Unmitigated)
    noise_model = _build_stage3_noise_model()
    noisy_backend = AerSimulator(noise_model=noise_model)
    pm_noisy = generate_preset_pass_manager(
        optimization_level=1, backend=noisy_backend, seed_transpiler=TRANSPILER_SEED
    )
    t_qc_noisy = pm_noisy.run(qc_meas)
    sampler_noisy = AerSamplerV2(
        default_shots=shots,
        seed=SIMULATOR_SEED,
        options={"backend_options": {"noise_model": noise_model}},
    )
    res_noisy = sampler_noisy.run([t_qc_noisy]).result()[0]
    counts_noisy = {str(k): int(v) for k, v in res_noisy.data.meas.get_counts().items()}
    p_noisy_unmit, pmf_noisy_unmit = _success_prob_from_counts(
        counts_noisy, n_qubits, shots, min_subscribers
    )

    # 3d. Readout Error Mitigation (tensor-product 2x2 per-qubit confusion matrix calibration)
    cal0 = QuantumCircuit(n_qubits)
    cal0.measure_all()
    cal1 = QuantumCircuit(n_qubits)
    cal1.x(range(n_qubits))
    cal1.measure_all()
    cal_res = sampler_noisy.run([pm_noisy.run(cal0), pm_noisy.run(cal1)], shots=shots).result()
    c0 = cal_res[0].data.meas.get_counts()
    c1 = cal_res[1].data.meas.get_counts()

    inv_mats = []
    per_qubit_cal = []
    for q in range(n_qubits):
        pos = n_qubits - 1 - q
        p0_given_0 = sum(v for k, v in c0.items() if k[pos] == "0") / shots
        p1_given_1 = sum(v for k, v in c1.items() if k[pos] == "1") / shots
        M = np.array([[p0_given_0, 1.0 - p1_given_1], [1.0 - p0_given_0, p1_given_1]], dtype=float)
        inv_mats.append(np.linalg.inv(M))
        per_qubit_cal.append(
            {
                "qubit_index": int(q),
                "customer_id": ids[q],
                "p0_given_0": float(p0_given_0),
                "p1_given_1": float(p1_given_1),
            }
        )

    probs_noisy_vec = np.zeros(1 << n_qubits, dtype=float)
    for bstr, cnt in counts_noisy.items():
        probs_noisy_vec[int(bstr, 2)] = cnt / shots

    probs_mit = probs_noisy_vec.reshape([2] * n_qubits).copy()
    for q in range(n_qubits):
        axis = n_qubits - 1 - q
        probs_mit = np.tensordot(inv_mats[q], probs_mit, axes=([1], [axis]))
        probs_mit = np.moveaxis(probs_mit, 0, axis)
    probs_mit = np.maximum(probs_mit.reshape(-1), 0.0)
    if np.sum(probs_mit) > 0:
        probs_mit /= np.sum(probs_mit)
    p_noisy_mit, pmf_noisy_mit = _success_prob_from_dist(probs_mit, n_qubits, min_subscribers)

    # Marginal qubit counts and P(1) checks from finite-shot ideal Aer
    marginal_ones_counts = []
    marginal_p1_shots = []
    for q in range(n_qubits):
        pos = n_qubits - 1 - q
        ones_q = int(sum(cnt for bstr, cnt in counts_shots.items() if bstr[pos] == "1"))
        marginal_ones_counts.append(ones_q)
        marginal_p1_shots.append(float(ones_q / shots))

    success_count_shots = int(
        sum(cnt for bstr, cnt in counts_shots.items() if bstr.replace(" ", "").count("1") >= min_subscribers)
    )
    success_count_noisy = int(
        sum(cnt for bstr, cnt in counts_noisy.items() if bstr.replace(" ", "").count("1") >= min_subscribers)
    )
    hw_counts_shots = {str(k): int(round(v * shots)) for k, v in pmf_shots.items()}

    errors = {
        "ideal_sv_minus_exact": float(p_ideal_sv - p_exact),
        "shots_minus_exact": float(p_shots - p_exact),
        "noisy_unmitigated_minus_exact": float(p_noisy_unmit - p_exact),
        "noisy_mitigated_minus_exact": float(p_noisy_mit - p_exact),
        "mc_100k_minus_exact": float(mc_100k["probability_at_least_k"] - p_exact),
        "mc_matched_minus_exact": float(mc_matched["probability_at_least_k"] - p_exact),
    }

    # Build top-level Stage 3 CSV (one row per selected customer using customer_id as key)
    result_df = pd.DataFrame(
        {
            "customer_id": ids,
            "stage1_rank": selected_df["rank"].to_numpy(dtype=int),
            "qubit_index": np.arange(n_qubits, dtype=int),
            "bitstring_position_msb0": [n_qubits - 1 - q for q in range(n_qubits)],
            "predicted_probability": probs,
            "ry_theta_rad": thetas,
            "shots": int(shots),
            "marginal_ones_count_shots": marginal_ones_counts,
            "marginal_zeros_count_shots": [int(shots - c) for c in marginal_ones_counts],
            "simulated_marginal_p1_shots": marginal_p1_shots,
            "quantum_success_count_shots": success_count_shots,
            "quantum_success_count_noisy_unmitigated": success_count_noisy,
            "hamming_weight_counts_shots_json": json.dumps(hw_counts_shots),
            "quantum_estimate_ideal_sv": p_ideal_sv,
            "quantum_estimate_shots": p_shots,
            "quantum_estimate_noisy_unmitigated": p_noisy_unmit,
            "quantum_estimate_noisy_mitigated": p_noisy_mit,
            "exact_probability": p_exact,
            "monte_carlo_100k_probability": mc_100k["probability_at_least_k"],
            "monte_carlo_matched_probability": mc_matched["probability_at_least_k"],
            "error_shots_vs_exact": errors["shots_minus_exact"],
            "error_noisy_unmit_vs_exact": errors["noisy_unmitigated_minus_exact"],
            "error_noisy_mit_vs_exact": errors["noisy_mitigated_minus_exact"],
            "error_mc_100k_vs_exact": errors["mc_100k_minus_exact"],
        }
    )

    qubit_mapping = [
        {
            "qubit_index": int(q),
            "bitstring_char_index_msb0": int(n_qubits - 1 - q),
            "customer_id": ids[q],
            "stage1_rank": int(selected_df.loc[q, "rank"]),
            "predicted_probability": float(probs[q]),
            "ry_theta_rad": float(thetas[q]),
            "simulated_marginal_p1_shots": marginal_p1_shots[q],
        }
        for q in range(n_qubits)
    ]

    return {
        "result_df": result_df,
        "selected_ids": ids,
        "bit_ordering_convention": "Qiskit little-endian: bitstring b of length 6 represents |q5 q4 q3 q2 q1 q0>, so qubit j is character index (5 - j).",
        "qubit_mapping": qubit_mapping,
        "shots": int(shots),
        "exact_poisson_binomial": exact_res,
        "monte_carlo_100k": mc_100k,
        "monte_carlo_matched_shots": mc_matched,
        "quantum_results": {
            "ideal_statevector": {"probability_at_least_3": p_ideal_sv, "pmf": pmf_ideal_sv},
            "noiseless_finite_shot": {
                "probability_at_least_3": p_shots,
                "pmf": pmf_shots,
                "measurement_counts": counts_shots,
            },
            "noisy_unmitigated": {
                "probability_at_least_3": p_noisy_unmit,
                "pmf": pmf_noisy_unmit,
                "measurement_counts": counts_noisy,
            },
            "noisy_readout_mitigated": {
                "probability_at_least_3": p_noisy_mit,
                "pmf": pmf_noisy_mit,
                "readout_calibration": per_qubit_cal,
            },
        },
        "errors": errors,
        "circuit_metrics_aer": circ_metrics_aer,
        "circuit_metrics_linear_backend": circ_metrics_hw,
        "circuit_qasm_text": str(qc_meas.draw(output="text")),
    }

