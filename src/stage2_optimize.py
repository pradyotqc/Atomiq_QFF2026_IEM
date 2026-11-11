"""Stage 2: Constrained Customer Selection via QUBO/Ising Quantum Optimization & Exact Benchmark."""

from __future__ import annotations

import itertools
import json
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate, real_amplitudes
from qiskit.primitives import StatevectorSampler
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.minimum_eigensolvers import QAOA, SamplingVQE
from qiskit_optimization.optimizers import COBYLA

from .config import (
    OPTIMIZER_SEED,
    SIMULATOR_SEED,
    STAGE2_BUDGET_C,
    STAGE2_CAPACITY_H,
    STAGE2_EXACT_K,
    STAGE2_QAOA_SHOTS,
)
from .metrics import extract_circuit_metrics


def _build_stage2_noise_model() -> NoiseModel:
    """Build realistic hardware-like depolarizing + readout error model for Aer validation."""
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(0.001, 1), ["rx", "ry", "rz", "sx", "x", "h", "p", "u"]
    )
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(0.01, 2), ["cx", "ecr", "cz", "rzz", "rxx", "ryy"]
    )
    readout = ReadoutError([[0.985, 0.015], [0.025, 0.975]])
    noise_model.add_all_qubit_readout_error(readout)
    return noise_model


def exact_brute_force_selection(
    candidates_df: pd.DataFrame,
    exact_k: int = STAGE2_EXACT_K,
    budget_limit: float = STAGE2_BUDGET_C,
    capacity_limit: float = STAGE2_CAPACITY_H,
) -> Dict[str, Any]:
    """Exhaustively enumerate all C(N, K) subsets (and check all 2^N bitstrings) for exact optimum."""
    n = len(candidates_df)
    values = candidates_df["expected_net_value"].to_numpy(dtype=float)
    costs = candidates_df["contact_cost"].to_numpy(dtype=float)
    caps = candidates_df["capacity_units"].to_numpy(dtype=float)
    ids = candidates_df["customer_id"].astype(str).tolist()

    feasible_records: List[Dict[str, Any]] = []
    for combo in itertools.combinations(range(n), exact_k):
        x = np.zeros(n, dtype=int)
        x[list(combo)] = 1
        total_cost = float(np.dot(costs, x))
        total_cap = float(np.dot(caps, x))
        total_val = float(np.dot(values, x))
        if total_cost <= budget_limit + 1e-9 and total_cap <= capacity_limit + 1e-9:
            feasible_records.append(
                {
                    "x": x.tolist(),
                    "bitstring_x0_to_x9": "".join(str(b) for b in x),
                    "selected_ids": [ids[i] for i in combo],
                    "objective_value": total_val,
                    "budget_total": total_cost,
                    "capacity_total": total_cap,
                }
            )

    if not feasible_records:
        raise RuntimeError("No feasible Stage 2 subset satisfies the constraints!")

    # Sort descending by objective_value, deterministic tie-breaker on selected_ids
    feasible_records.sort(key=lambda r: (-r["objective_value"], tuple(r["selected_ids"])))
    best = feasible_records[0]

    return {
        "total_2pow_n": int(2**n),
        "total_combinations_n_choose_k": int(len(list(itertools.combinations(range(n), exact_k)))),
        "feasible_count": int(len(feasible_records)),
        "optimal": best,
        "top_5_feasible": feasible_records[:5],
    }


def _verify_ising_ground_state_vectorized(
    ising_op: SparsePauliOp,
    offset: float,
    num_decision_vars: int,
    exact_x: List[int],
    exact_obj_max: float,
) -> Dict[str, Any]:
    """Verify via exact vectorized diagonal scan over 2^n_qubits that QUBO ground state matches exact."""
    n_qubits = ising_op.num_qubits
    # Build integer bit table [2^n_qubits, n_qubits] in little-endian qubit order (qubit 0 = LSB)
    states = np.arange(1 << n_qubits, dtype=np.uint32)
    z_table = np.empty((1 << n_qubits, n_qubits), dtype=np.int8)
    for q in range(n_qubits):
        bits_q = (states >> q) & 1
        z_table[:, q] = 1 - 2 * bits_q.astype(np.int8)

    energies = np.full(1 << n_qubits, float(offset), dtype=np.float64)
    for pauli, coeff in zip(ising_op.paulis, ising_op.coeffs):
        c = float(np.real(coeff))
        z_mask = pauli.z
        active_qubits = np.where(z_mask)[0]
        if len(active_qubits) == 0:
            energies += c
        elif len(active_qubits) == 1:
            energies += c * z_table[:, active_qubits[0]].astype(np.float64)
        else:
            term = z_table[:, active_qubits[0]].copy()
            for q in active_qubits[1:]:
                term *= z_table[:, q]
            energies += c * term.astype(np.float64)

    min_idx = int(np.argmin(energies))
    min_energy = float(energies[min_idx])
    ground_x = [int((min_idx >> q) & 1) for q in range(num_decision_vars)]

    return {
        "qubo_num_qubits": int(n_qubits),
        "qubo_exact_min_energy": min_energy,
        "expected_negated_max_objective": float(-exact_obj_max),
        "ground_state_matches_exact": bool(
            abs(min_energy - (-exact_obj_max)) < 1e-6 and ground_x == exact_x
        ),
        "qubo_ground_decision_x": ground_x,
    }


def _decode_solution_samples(
    samples: List[Any],
    ids: List[str],
    costs: np.ndarray,
    caps: np.ndarray,
    values: np.ndarray,
    exact_k: int,
    budget_limit: float,
    capacity_limit: float,
    max_items: int = 8,
) -> Tuple[Dict[str, Any] | None, List[Dict[str, Any]]]:
    """Decode Qiskit Optimization samples into human-readable bitstrings and verify feasibility."""
    decoded_list: List[Dict[str, Any]] = []
    best_feasible: Dict[str, Any] | None = None

    for s in samples:
        x_dec = [int(round(v)) for v in s.x[: len(ids)]]
        n_sel = int(sum(x_dec))
        b_tot = float(np.dot(costs, x_dec))
        c_tot = float(np.dot(caps, x_dec))
        obj_val = float(np.dot(values, x_dec))
        feasible = (
            n_sel == exact_k
            and b_tot <= budget_limit + 1e-9
            and c_tot <= capacity_limit + 1e-9
        )
        entry = {
            "bitstring_x0_to_x9": "".join(str(b) for b in x_dec),
            "selected_ids": [ids[i] for i, b in enumerate(x_dec) if b == 1],
            "count": n_sel,
            "budget_total": b_tot,
            "capacity_total": c_tot,
            "objective_value": obj_val,
            "probability": float(s.probability),
            "feasible": bool(feasible),
        }
        decoded_list.append(entry)
        if feasible:
            if best_feasible is None or obj_val > best_feasible["objective_value"] + 1e-9:
                best_feasible = entry

    # Sort decoded list by feasibility desc, objective desc, probability desc
    decoded_list.sort(
        key=lambda d: (-int(d["feasible"]), -d["objective_value"], -d["probability"])
    )
    return best_feasible, decoded_list[:max_items]


def _run_qubo_solvers(
    candidates_df: pd.DataFrame,
    exact_res: Dict[str, Any],
    exact_k: int,
    budget_limit: float,
    capacity_limit: float,
) -> Dict[str, Any]:
    """Formulate QuadraticProgram -> QUBO with slack bits, verify ground state, run QAOA & CVaR-SamplingVQE."""
    n = len(candidates_df)
    ids = candidates_df["customer_id"].astype(str).tolist()
    values = candidates_df["expected_net_value"].to_numpy(dtype=float)
    costs = candidates_df["contact_cost"].to_numpy(dtype=float)
    caps = candidates_df["capacity_units"].to_numpy(dtype=float)

    qp = QuadraticProgram("TermDepositSelection")
    for i in range(n):
        qp.binary_var(name=f"x_{i}")

    qp.maximize(linear={f"x_{i}": float(values[i]) for i in range(n)})
    qp.linear_constraint(
        linear={f"x_{i}": 1.0 for i in range(n)}, sense="==", rhs=float(exact_k), name="exact_k"
    )
    qp.linear_constraint(
        linear={f"x_{i}": float(costs[i]) for i in range(n)},
        sense="<=",
        rhs=float(budget_limit),
        name="budget",
    )
    qp.linear_constraint(
        linear={f"x_{i}": float(caps[i]) for i in range(n)},
        sense="<=",
        rhs=float(capacity_limit),
        name="capacity",
    )

    # Penalty guaranteed strictly larger than sum of positive objective terms
    penalty = float(np.ceil(np.sum(np.abs(values)) + 5.0))
    converter = QuadraticProgramToQubo(penalty=penalty)
    qubo = converter.convert(qp)
    ising_op, offset = qubo.to_ising()
    num_qubo_qubits = qubo.get_num_binary_vars()

    exact_opt = exact_res["optimal"]
    qubo_ground_check = _verify_ising_ground_state_vectorized(
        ising_op,
        offset,
        n,
        exact_opt["x"],
        exact_opt["objective_value"],
    )

    aer_sv_backend = AerSimulator(method="statevector", max_parallel_threads=8)
    pm_sv = generate_preset_pass_manager(
        optimization_level=1, backend=aer_sv_backend, seed_transpiler=SIMULATOR_SEED
    )
    sampler = AerSamplerV2(
        default_shots=STAGE2_QAOA_SHOTS,
        seed=SIMULATOR_SEED,
        options={"backend_options": {"method": "statevector", "max_parallel_threads": 8}},
    )

    # --- Solver 1: 20-qubit QUBO with CVaR-QAOA (p=1) ---
    qaoa_history: List[float] = []

    def qaoa_cb(eval_count: int, params: np.ndarray, mean: float, meta: dict) -> None:
        qaoa_history.append(float(np.real(mean)))

    qaoa_init = np.array([0.35, 0.25], dtype=float)
    qaoa_solver = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=60),
        reps=1,
        aggregation=0.15,
        initial_point=qaoa_init,
        callback=qaoa_cb,
        pass_manager=pm_sv,
    )
    meo_qaoa = MinimumEigenOptimizer(qaoa_solver, converters=converter)
    qaoa_result = meo_qaoa.solve(qp)
    qaoa_best_feasible, qaoa_top_samples = _decode_solution_samples(
        qaoa_result.samples, ids, costs, caps, values, exact_k, budget_limit, capacity_limit
    )

    # Extract QAOA circuit transpilation metrics
    qaoa_bound = qaoa_solver.ansatz.assign_parameters(qaoa_result.min_eigen_solver_result.optimal_point)
    qaoa_transpiled = transpile(
        qaoa_bound, backend=aer_sv_backend, optimization_level=1, seed_transpiler=SIMULATOR_SEED
    )
    qaoa_circ_metrics = extract_circuit_metrics(qaoa_bound, qaoa_transpiled)

    # --- Solver 2: 20-qubit QUBO with CVaR-SamplingVQE ---
    vqe_ansatz = real_amplitudes(num_qubits=num_qubo_qubits, reps=1, entanglement="linear")
    vqe_ansatz_isa = pm_sv.run(vqe_ansatz)
    rng = np.random.default_rng(OPTIMIZER_SEED)
    vqe_init = rng.uniform(-0.2, 0.2, size=vqe_ansatz_isa.num_parameters)
    vqe_history: List[float] = []

    def vqe_cb(eval_count: int, params: np.ndarray, mean: float, meta: dict) -> None:
        vqe_history.append(float(np.real(mean)))

    vqe_solver = SamplingVQE(
        sampler=sampler,
        ansatz=vqe_ansatz_isa,
        optimizer=COBYLA(maxiter=200),
        aggregation=0.15,
        initial_point=vqe_init,
        callback=vqe_cb,
        pass_manager=pm_sv,
    )
    meo_vqe = MinimumEigenOptimizer(vqe_solver, converters=converter)
    vqe_result = meo_vqe.solve(qp)
    vqe_best_feasible, vqe_top_samples = _decode_solution_samples(
        vqe_result.samples, ids, costs, caps, values, exact_k, budget_limit, capacity_limit
    )

    vqe_bound = vqe_ansatz_isa.assign_parameters(vqe_result.min_eigen_solver_result.optimal_point)
    vqe_circ_metrics = extract_circuit_metrics(vqe_ansatz.assign_parameters(vqe_result.min_eigen_solver_result.optimal_point), vqe_bound)

    return {
        "qubo_details": {
            "num_decision_variables": int(n),
            "num_slack_variables": int(num_qubo_qubits - n),
            "total_qubo_qubits": int(num_qubo_qubits),
            "qubo_variable_names": [v.name for v in qubo.variables],
            "penalty_weight": penalty,
            "ising_num_pauli_terms": int(len(ising_op)),
            "ising_offset": float(offset),
            "ground_state_verification": qubo_ground_check,
        },
        "qubo_cvar_qaoa": {
            "reps": 1,
            "cvar_alpha": 0.15,
            "shots": STAGE2_QAOA_SHOTS,
            "optimal_parameters": [
                float(x) for x in qaoa_result.min_eigen_solver_result.optimal_point
            ],
            "num_evaluations": int(len(qaoa_history)),
            "cvar_energy_history": qaoa_history,
            "best_feasible_selection": qaoa_best_feasible,
            "top_measured_bitstrings": qaoa_top_samples,
            "circuit_metrics": qaoa_circ_metrics,
        },
        "qubo_cvar_sampling_vqe": {
            "ansatz": "RealAmplitudes(reps=1, entanglement='linear')",
            "cvar_alpha": 0.15,
            "shots": STAGE2_QAOA_SHOTS,
            "num_parameters": int(vqe_ansatz.num_parameters),
            "optimal_parameters": [
                float(x) for x in vqe_result.min_eigen_solver_result.optimal_point
            ],
            "num_evaluations": int(len(vqe_history)),
            "cvar_energy_history": vqe_history,
            "best_feasible_selection": vqe_best_feasible,
            "top_measured_bitstrings": vqe_top_samples,
            "circuit_metrics": vqe_circ_metrics,
        },
    }


def _run_10qubit_constrained_qaoa_benchmark(
    candidates_df: pd.DataFrame,
    exact_res: Dict[str, Any],
    exact_k: int,
    budget_limit: float,
    capacity_limit: float,
) -> Dict[str, Any]:
    """10-qubit Dicke-state |D_10^6> + Ring XY-Mixer QAOA across Ideal, Finite-Shot, Noisy & Mitigated."""
    n = len(candidates_df)
    ids = candidates_df["customer_id"].astype(str).tolist()
    values = candidates_df["expected_net_value"].to_numpy(dtype=float)
    costs = candidates_df["contact_cost"].to_numpy(dtype=float)
    caps = candidates_df["capacity_units"].to_numpy(dtype=float)

    # Build uniform Dicke initial state |D_10^6> over all C(10, 6) = 210 Hamming-weight-6 basis states
    dicke_vec = np.zeros(1 << n, dtype=complex)
    valid_weight_indices = [idx for idx in range(1 << n) if idx.bit_count() == exact_k]
    dicke_vec[valid_weight_indices] = 1.0 / np.sqrt(len(valid_weight_indices))

    # Build diagonal cost energies E(x) over 2^10 states: -obj(x) + penalty on any constraint violation
    penalty = float(np.ceil(np.sum(np.abs(values)) + 10.0))
    energies = np.zeros(1 << n, dtype=float)
    feasible_mask = np.zeros(1 << n, dtype=bool)
    objectives = np.zeros(1 << n, dtype=float)

    for idx in range(1 << n):
        x = np.array([(idx >> q) & 1 for q in range(n)], dtype=float)
        obj = float(np.dot(values, x))
        objectives[idx] = obj
        b_tot = float(np.dot(costs, x))
        c_tot = float(np.dot(caps, x))
        viol = abs(int(np.sum(x)) - exact_k) + max(0.0, b_tot - budget_limit) + max(0.0, c_tot - capacity_limit)
        if viol == 0.0:
            feasible_mask[idx] = True
            energies[idx] = -obj
        else:
            energies[idx] = -obj + penalty * viol

    # Brickwork Ring XY mixer: exp(-i * dt/2 * (X_u X_v + Y_u Y_v)) = RXX(dt) RYY(dt)
    # Every two-qubit XY gate preserves total Hamming weight |D_10^6> identically.
    even_edges = [(q, (q + 1) % n) for q in range(0, n, 2)]
    odd_edges = [(q, (q + 1) % n) for q in range(1, n, 2)]
    trotter_steps = 2

    def build_xy_qaoa_circuit(gammas: np.ndarray, betas: np.ndarray, add_measure: bool = False) -> QuantumCircuit:
        qc = QuantumCircuit(n)
        qc.initialize(dicke_vec, range(n))
        for gamma, beta in zip(gammas, betas):
            phases = np.exp(-1j * gamma * energies).tolist()
            qc.append(DiagonalGate(phases), range(n))
            dt = float(beta) / float(trotter_steps)
            for _ in range(trotter_steps):
                for u, v in even_edges:
                    qc.rxx(dt, u, v)
                    qc.ryy(dt, u, v)
                for u, v in odd_edges:
                    qc.rxx(dt, u, v)
                    qc.ryy(dt, u, v)
        if add_measure:
            qc.measure_all()
        return qc

    reps = 2
    alpha = 0.15

    def cvar_from_probs(probs: np.ndarray) -> float:
        order = np.argsort(energies)
        cum = 0.0
        weighted = 0.0
        for idx in order:
            p = float(probs[idx])
            if p <= 0:
                continue
            take = min(p, alpha - cum)
            if take > 0:
                weighted += take * energies[idx]
                cum += take
            if cum >= alpha - 1e-12:
                break
        return float(weighted / max(cum, 1e-12))

    # Optimize parameters on exact ideal statevector
    history: List[float] = []

    def obj_fn(theta: np.ndarray) -> float:
        qc = build_xy_qaoa_circuit(theta[:reps], theta[reps:], add_measure=False)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities()
        val = cvar_from_probs(probs)
        history.append(val)
        return val

    init_theta = np.array([0.25, 0.45, 0.35, 0.25], dtype=float)
    opt_res = minimize(obj_fn, init_theta, method="COBYLA", options={"maxiter": 180, "rhobeg": 0.35})
    opt_theta = opt_res.x

    # 1) Ideal Statevector evaluation for XY-QAOA
    qc_ideal_nom = build_xy_qaoa_circuit(opt_theta[:reps], opt_theta[reps:], add_measure=False)
    probs_ideal = Statevector.from_instruction(qc_ideal_nom).probabilities()

    # 2) Noiseless Finite-Shot evaluation for XY-QAOA
    qc_meas = build_xy_qaoa_circuit(opt_theta[:reps], opt_theta[reps:], add_measure=True)
    shots = STAGE2_QAOA_SHOTS
    sv_sampler = StatevectorSampler(default_shots=shots, seed=SIMULATOR_SEED)
    res_shots = sv_sampler.run([qc_meas]).result()[0]
    counts_shots = res_shots.data.meas.get_counts()
    probs_shots = np.zeros(1 << n, dtype=float)
    for bstr, cnt in counts_shots.items():
        probs_shots[int(bstr, 2)] = cnt / shots

    # 3) Hardware-efficient 10-qubit CVaR-VQE circuit for realistic Noisy Aer + Readout Mitigation
    hw_ansatz = real_amplitudes(num_qubits=n, reps=1, entanglement="linear")
    rng_hw = np.random.default_rng(OPTIMIZER_SEED)
    hw_init = rng_hw.uniform(-0.3, 0.3, size=hw_ansatz.num_parameters)

    def hw_obj_fn(theta: np.ndarray) -> float:
        sv = Statevector.from_instruction(hw_ansatz.assign_parameters(theta))
        return cvar_from_probs(sv.probabilities())

    hw_opt = minimize(hw_obj_fn, hw_init, method="COBYLA", options={"maxiter": 260, "rhobeg": 0.4})
    qc_hw_nom = hw_ansatz.assign_parameters(hw_opt.x)
    probs_hw_ideal = Statevector.from_instruction(qc_hw_nom).probabilities()

    qc_hw_meas = qc_hw_nom.copy()
    qc_hw_meas.measure_all()

    noise_model = _build_stage2_noise_model()
    noisy_backend = AerSimulator(noise_model=noise_model)
    pm_noisy = generate_preset_pass_manager(
        optimization_level=1, backend=noisy_backend, seed_transpiler=SIMULATOR_SEED
    )
    t_qc_hw_noisy = pm_noisy.run(qc_hw_meas)

    # Also transpile onto linear-chain GenericBackendV2 to capture realistic SWAP & 2Q overhead
    linear_backend = GenericBackendV2(
        num_qubits=n,
        coupling_map=[[i, i + 1] for i in range(n - 1)],
        seed=SIMULATOR_SEED,
    )
    t_qc_hw_linear = transpile(qc_hw_meas, backend=linear_backend, optimization_level=1, seed_transpiler=SIMULATOR_SEED)
    circ_metrics_hw_aer = extract_circuit_metrics(qc_hw_meas, t_qc_hw_noisy)
    circ_metrics_hw_linear = extract_circuit_metrics(qc_hw_meas, t_qc_hw_linear)

    sampler_hw_ideal = AerSamplerV2(default_shots=shots, seed=SIMULATOR_SEED)
    t_qc_hw_ideal = transpile(qc_hw_meas, backend=AerSimulator(), optimization_level=1, seed_transpiler=SIMULATOR_SEED)
    res_hw_shots = sampler_hw_ideal.run([t_qc_hw_ideal]).result()[0]
    counts_hw_shots = res_hw_shots.data.meas.get_counts()
    probs_hw_shots = np.zeros(1 << n, dtype=float)
    for bstr, cnt in counts_hw_shots.items():
        probs_hw_shots[int(bstr, 2)] = cnt / shots

    sampler_noisy = AerSamplerV2(
        default_shots=shots,
        seed=SIMULATOR_SEED,
        options={"backend_options": {"noise_model": noise_model}},
    )
    res_noisy = sampler_noisy.run([t_qc_hw_noisy]).result()[0]
    counts_noisy = res_noisy.data.meas.get_counts()
    probs_noisy = np.zeros(1 << n, dtype=float)
    for bstr, cnt in counts_noisy.items():
        probs_noisy[int(bstr, 2)] = cnt / shots

    # Readout Error Mitigation (calibrate 2x2 per-qubit confusion matrices via |0...0> and |1...1>)
    cal0 = QuantumCircuit(n)
    cal0.measure_all()
    cal1 = QuantumCircuit(n)
    cal1.x(range(n))
    cal1.measure_all()
    cal_res = sampler_noisy.run([pm_noisy.run(cal0), pm_noisy.run(cal1)], shots=shots).result()
    c0 = cal_res[0].data.meas.get_counts()
    c1 = cal_res[1].data.meas.get_counts()

    inv_mats = []
    for q in range(n):
        pos = n - 1 - q
        p0_given_0 = sum(v for k, v in c0.items() if k[pos] == "0") / shots
        p1_given_1 = sum(v for k, v in c1.items() if k[pos] == "1") / shots
        M = np.array([[p0_given_0, 1.0 - p1_given_1], [1.0 - p0_given_0, p1_given_1]], dtype=float)
        inv_mats.append(np.linalg.inv(M))

    probs_mit = probs_noisy.reshape([2] * n).copy()
    for q in range(n):
        axis = n - 1 - q
        probs_mit = np.tensordot(inv_mats[q], probs_mit, axes=([1], [axis]))
        probs_mit = np.moveaxis(probs_mit, 0, axis)
    probs_mit = np.maximum(probs_mit.reshape(-1), 0.0)
    if np.sum(probs_mit) > 0:
        probs_mit /= np.sum(probs_mit)

    exact_opt = exact_res["optimal"]
    opt_idx = sum(b << q for q, b in enumerate(exact_opt["x"]))

    def summarize_dist(probs: np.ndarray) -> Dict[str, Any]:
        feas_prob = float(np.sum(probs[feasible_mask]))
        opt_prob = float(probs[opt_idx])
        top_indices = np.argsort(-probs)[:8]
        top_bits = []
        best_feas_idx = None
        for idx in np.argsort(-probs):
            if feasible_mask[idx]:
                best_feas_idx = int(idx)
                break
        for idx in top_indices:
            x_vec = [(int(idx) >> q) & 1 for q in range(n)]
            top_bits.append(
                {
                    "bitstring_x0_to_x9": "".join(str(b) for b in x_vec),
                    "selected_ids": [ids[i] for i, b in enumerate(x_vec) if b == 1],
                    "objective_value": float(objectives[idx]),
                    "feasible": bool(feasible_mask[idx]),
                    "probability": float(probs[idx]),
                }
            )
        best_sel = None
        if best_feas_idx is not None:
            x_vec = [(best_feas_idx >> q) & 1 for q in range(n)]
            best_sel = {
                "bitstring_x0_to_x9": "".join(str(b) for b in x_vec),
                "selected_ids": [ids[i] for i, b in enumerate(x_vec) if b == 1],
                "objective_value": float(objectives[best_feas_idx]),
                "budget_total": float(np.dot(costs, x_vec)),
                "capacity_total": float(np.dot(caps, x_vec)),
                "probability": float(probs[best_feas_idx]),
                "feasible": True,
            }
        return {
            "feasible_probability_mass": feas_prob,
            "exact_optimum_probability": opt_prob,
            "best_feasible_selection": best_sel,
            "top_bitstrings": top_bits,
        }

    return {
        "n_qubits": int(n),
        "reps": int(reps),
        "cvar_alpha": float(alpha),
        "shots": int(shots),
        "optimal_parameters": [float(x) for x in opt_theta],
        "cvar_energy_history": history,
        "ideal_statevector": summarize_dist(probs_ideal),
        "noiseless_finite_shot": summarize_dist(probs_shots),
        "hardware_efficient_10q_vqe": {
            "optimal_parameters": [float(x) for x in hw_opt.x],
            "circuit_metrics_aer": circ_metrics_hw_aer,
            "circuit_metrics_linear_backend": circ_metrics_hw_linear,
            "ideal_statevector": summarize_dist(probs_hw_ideal),
            "noiseless_finite_shot": summarize_dist(probs_hw_shots),
            "noisy_unmitigated": summarize_dist(probs_noisy),
            "noisy_readout_mitigated": summarize_dist(probs_mit),
        },
    }


def run_stage2(
    stage1_result_df: pd.DataFrame,
    stage2_data_df: pd.DataFrame,
    exact_k: int = STAGE2_EXACT_K,
    budget_limit: float = STAGE2_BUDGET_C,
    capacity_limit: float = STAGE2_CAPACITY_H,
) -> Dict[str, Any]:
    """Run Stage 2 selection using Stage 1 top-10 candidates, exact brute-force, and quantum solvers."""
    top10_df = (
        stage1_result_df[stage1_result_df["top_10_flag"] == 1]
        .sort_values(by=["rank", "customer_id"], ascending=[True, True])
        .reset_index(drop=True)
    )
    if len(top10_df) != 10:
        raise ValueError(f"Expected exactly 10 candidates from Stage 1, got {len(top10_df)}")

    candidates_df = top10_df.merge(stage2_data_df, on="customer_id", how="left")
    if candidates_df[["contact_cost", "expected_deposit_value", "capacity_units"]].isna().any().any():
        raise ValueError("Missing Stage2_Data entries for one or more top-10 candidates.")

    candidates_df["expected_net_value"] = (
        candidates_df["predicted_probability"] * candidates_df["expected_deposit_value"]
        - candidates_df["contact_cost"]
    )

    exact_res = exact_brute_force_selection(
        candidates_df, exact_k=exact_k, budget_limit=budget_limit, capacity_limit=capacity_limit
    )
    qubo_res = _run_qubo_solvers(
        candidates_df, exact_res, exact_k=exact_k, budget_limit=budget_limit, capacity_limit=capacity_limit
    )
    xy_qaoa_res = _run_10qubit_constrained_qaoa_benchmark(
        candidates_df, exact_res, exact_k=exact_k, budget_limit=budget_limit, capacity_limit=capacity_limit
    )

    # Choose the primary quantum selection from the best feasible quantum solver result
    # Prefer full QUBO CVaR-SamplingVQE / CVaR-QAOA if optimal & feasible, else constrained XY-QAOA
    candidate_solvers = [
        ("QUBO_CVaR_SamplingVQE", qubo_res["qubo_cvar_sampling_vqe"]["best_feasible_selection"]),
        ("QUBO_CVaR_QAOA", qubo_res["qubo_cvar_qaoa"]["best_feasible_selection"]),
        ("XY_Mixer_QAOA_10q", xy_qaoa_res["noiseless_finite_shot"]["best_feasible_selection"]),
    ]
    valid_solvers = [(name, sel) for name, sel in candidate_solvers if sel is not None and sel["feasible"]]
    if not valid_solvers:
        raise RuntimeError("No quantum solver returned a feasible 6-customer selection!")

    # Pick highest objective among feasible quantum solvers
    primary_solver_name, primary_sel = max(
        valid_solvers, key=lambda item: item[1]["objective_value"]
    )
    exact_opt_val = exact_res["optimal"]["objective_value"]
    optimality_gap = float(exact_opt_val - primary_sel["objective_value"])

    selected_set = set(primary_sel["selected_ids"])
    candidates_df["selected_flag"] = candidates_df["customer_id"].isin(selected_set).astype(int)
    candidates_df["classical_exact_selected_flag"] = (
        candidates_df["customer_id"].isin(set(exact_res["optimal"]["selected_ids"])).astype(int)
    )
    candidates_df["primary_quantum_solver"] = primary_solver_name
    candidates_df["selected_budget_total"] = float(primary_sel["budget_total"])
    candidates_df["selected_capacity_total"] = float(primary_sel["capacity_total"])
    candidates_df["quantum_objective_value"] = float(primary_sel["objective_value"])
    candidates_df["classical_exact_objective"] = float(exact_opt_val)
    candidates_df["optimality_gap"] = optimality_gap
    candidates_df["best_bitstring_x0_to_x9"] = primary_sel["bitstring_x0_to_x9"]
    candidates_df["top3_qaoa_bitstrings"] = ";".join(
        item["bitstring_x0_to_x9"]
        for item in qubo_res["qubo_cvar_qaoa"]["top_measured_bitstrings"][:3]
    )
    vqe_sel = qubo_res["qubo_cvar_sampling_vqe"]["best_feasible_selection"]
    candidates_df["sampling_vqe_bitstring_x0_to_x9"] = (
        vqe_sel["bitstring_x0_to_x9"] if vqe_sel is not None else ""
    )
    hw_sel = xy_qaoa_res["hardware_efficient_10q_vqe"]["noiseless_finite_shot"][
        "best_feasible_selection"
    ]
    candidates_df["hw_vqe_10q_bitstring_x0_to_x9"] = (
        hw_sel["bitstring_x0_to_x9"] if hw_sel is not None else ""
    )

    result_cols = [
        "customer_id",
        "rank",
        "predicted_probability",
        "contact_cost",
        "expected_deposit_value",
        "capacity_units",
        "customer_segment",
        "expected_net_value",
        "selected_flag",
        "classical_exact_selected_flag",
        "primary_quantum_solver",
        "selected_budget_total",
        "selected_capacity_total",
        "quantum_objective_value",
        "classical_exact_objective",
        "optimality_gap",
        "best_bitstring_x0_to_x9",
        "top3_qaoa_bitstrings",
        "sampling_vqe_bitstring_x0_to_x9",
        "hw_vqe_10q_bitstring_x0_to_x9",
    ]
    result_df = candidates_df[result_cols].copy()

    return {
        "primary_solver_name": primary_solver_name,
        "primary_selection": primary_sel,
        "optimality_gap": optimality_gap,
        "result_df": result_df,
        "exact_benchmark": exact_res,
        "qubo_solvers": qubo_res,
        "xy_qaoa_10q_benchmark": xy_qaoa_res,
    }
