"""Stage 1: Leakage-free Classical & Quantum/Hybrid Subscription Prediction."""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from qiskit import transpile
from qiskit.circuit.library import pauli_feature_map, real_amplitudes
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute

from .config import CV_N_SPLITS, CV_SPLIT_SEED, MODEL_SEED, SIMULATOR_SEED, TOP_K_CANDIDATES
from .metrics import compute_stage1_metrics, extract_circuit_metrics


MONTH_TO_INDEX = {
    "jan": 0, "feb": 1, "mar": 2, "apr": 3, "may": 4, "jun": 5,
    "jul": 6, "aug": 7, "sep": 8, "oct": 9, "nov": 10, "dec": 11,
}

NUMERIC_COLS = [
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous",
    "was_previously_contacted",
    "pdays_clean",
    "log1p_pdays",
    "signed_log1p_balance",
    "log1p_campaign",
    "log1p_previous",
    "has_prev_success",
    "has_prev_failure",
    "has_known_contact",
    "any_loan",
    "balance_per_age",
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
]

CATEGORICAL_COLS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]


def _zero_centered_pauli_map(x: np.ndarray) -> Any:
    """Zero-centered Pauli phase mapping: x_i for 1-qubit, x_i * x_j for 2-qubit interactions."""
    return x[0] if len(x) == 1 else x[0] * x[1]


def _build_zero_centered_feature_map(n_qubits: int, reps: int = 1):
    """Build hardware-friendly Z+ZZ Pauli feature map without Qiskit's (pi - x) offset."""
    return pauli_feature_map(
        feature_dimension=n_qubits,
        reps=reps,
        paulis=["Z", "ZZ"],
        entanglement="linear",
        alpha=1.0,
        data_map_func=_zero_centered_pauli_map,
    )


def _compute_feature_state_matrix(feature_map: Any, x_angles: np.ndarray) -> np.ndarray:
    """Evaluate exact Qiskit Statevectors |phi(x_i)> as [N, 2^n] complex array."""
    dim = 1 << feature_map.num_qubits
    phi = np.empty((len(x_angles), dim), dtype=complex)
    for i, row in enumerate(x_angles):
        phi[i, :] = Statevector.from_instruction(feature_map.assign_parameters(row)).data
    return phi


class BankFeatureEngineer(BaseEstimator, TransformerMixin):
    """Deterministic row-wise feature engineering (zero cross-row statistics)."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "BankFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        pdays = pd.to_numeric(df["pdays"], errors="coerce").fillna(-1.0)
        balance = pd.to_numeric(df["balance"], errors="coerce").fillna(0.0)
        campaign = pd.to_numeric(df["campaign"], errors="coerce").fillna(1.0)
        previous = pd.to_numeric(df["previous"], errors="coerce").fillna(0.0)
        age = pd.to_numeric(df["age"], errors="coerce").fillna(40.0)
        day = pd.to_numeric(df["day"], errors="coerce").fillna(15.0)

        df["was_previously_contacted"] = (pdays >= 0).astype(float)
        df["pdays_clean"] = np.where(pdays >= 0, pdays, 0.0)
        df["log1p_pdays"] = np.log1p(df["pdays_clean"])
        df["signed_log1p_balance"] = np.sign(balance) * np.log1p(np.abs(balance))
        df["log1p_campaign"] = np.log1p(np.maximum(campaign, 0.0))
        df["log1p_previous"] = np.log1p(np.maximum(previous, 0.0))
        df["has_prev_success"] = (df["poutcome"].astype(str) == "success").astype(float)
        df["has_prev_failure"] = (df["poutcome"].astype(str) == "failure").astype(float)
        df["has_known_contact"] = (df["contact"].astype(str) != "unknown").astype(float)
        df["any_loan"] = (
            (df["housing"].astype(str) == "yes") | (df["loan"].astype(str) == "yes")
        ).astype(float)
        df["balance_per_age"] = df["signed_log1p_balance"] / np.maximum(age, 18.0)

        month_idx = df["month"].astype(str).str.lower().map(MONTH_TO_INDEX).fillna(4.0)
        df["month_sin"] = np.sin(2.0 * np.pi * month_idx / 12.0)
        df["month_cos"] = np.cos(2.0 * np.pi * month_idx / 12.0)
        df["day_sin"] = np.sin(2.0 * np.pi * (day - 1.0) / 31.0)
        df["day_cos"] = np.cos(2.0 * np.pi * (day - 1.0) / 31.0)
        return df


def build_tabular_preprocessor() -> Pipeline:
    """Build leakage-free preprocessor fit strictly on training data."""
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    col_tf = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC_COLS),
            ("cat", cat_pipe, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )
    return Pipeline([("engineer", BankFeatureEngineer()), ("columns", col_tf)])


class QuantumAngleCompressor(BaseEstimator, TransformerMixin):
    """Select/project features to `n_qubits` and apply smooth bandwidth-controlled angle scaling."""

    def __init__(
        self, n_qubits: int = 6, bandwidth: float = 0.7, mode: str = "select_k"
    ) -> None:
        self.n_qubits = n_qubits
        self.bandwidth = bandwidth
        self.mode = mode

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumAngleCompressor":
        y_arr = np.asarray(y, dtype=int)
        if self.mode == "select_k":
            k = min(self.n_qubits, X.shape[1])
            self.selector_ = SelectKBest(f_classif, k=k).fit(X, y_arr)
            Z = self.selector_.transform(X)
            self.scaler_ = StandardScaler().fit(Z)
        elif self.mode == "pls_weighted":
            n_comp = min(self.n_qubits, X.shape[1], max(1, X.shape[0] - 1))
            self.pls_ = PLSRegression(n_components=n_comp).fit(X, y_arr.astype(float))
            Z = self.pls_.transform(X)
            self.global_scale_ = float(np.std(Z[:, 0]) + 1e-8)
        else:
            raise ValueError(f"Unsupported QuantumAngleCompressor mode: {self.mode}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mode == "select_k":
            Z = self.scaler_.transform(self.selector_.transform(X))
            angles = self.bandwidth * np.tanh(Z / 2.0)
        else:
            Z = self.pls_.transform(X)
            angles = self.bandwidth * np.tanh(Z / (2.0 * self.global_scale_))
        if angles.shape[1] < self.n_qubits:
            pad = np.zeros((angles.shape[0], self.n_qubits - angles.shape[1]))
            angles = np.hstack([angles, pad])
        return angles


class QuantumKernelLogRegClassifier(BaseEstimator, ClassifierMixin):
    """Exact RKHS Quantum Fidelity Kernel Logistic Regression using spectral kernel whitening."""

    def __init__(
        self,
        n_qubits: int = 6,
        reps: int = 1,
        bandwidth: float = 0.7,
        C: float = 2.0,
        projection_mode: str = "select_k",
        random_state: int = MODEL_SEED,
    ) -> None:
        self.n_qubits = n_qubits
        self.reps = reps
        self.bandwidth = bandwidth
        self.C = C
        self.projection_mode = projection_mode
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumKernelLogRegClassifier":
        y_arr = np.asarray(y, dtype=int)
        self.classes_ = np.array([0, 1])
        self.compressor_ = QuantumAngleCompressor(
            n_qubits=self.n_qubits, bandwidth=self.bandwidth, mode=self.projection_mode
        )
        self.X_train_angles_ = self.compressor_.fit_transform(X, y_arr)
        self.feature_map_ = _build_zero_centered_feature_map(self.n_qubits, reps=self.reps)
        self.phi_train_ = _compute_feature_state_matrix(self.feature_map_, self.X_train_angles_)
        K_train = np.abs(self.phi_train_ @ self.phi_train_.conj().T) ** 2

        # Spectral RKHS feature decomposition: K = V Lambda V^T -> F = K V Lambda^{-1/2}
        eigvals, eigvecs = np.linalg.eigh(K_train)
        keep = eigvals > 1e-8
        self.V_inv_sqrt_ = eigvecs[:, keep] / np.sqrt(eigvals[keep])
        F_train = K_train @ self.V_inv_sqrt_

        self.logreg_ = LogisticRegression(
            C=self.C, max_iter=2000, random_state=self.random_state
        )
        self.logreg_.fit(F_train, y_arr)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_angles = self.compressor_.transform(X)
        phi_eval = _compute_feature_state_matrix(self.feature_map_, X_angles)
        K_eval = np.abs(phi_eval @ self.phi_train_.conj().T) ** 2
        F_eval = K_eval @ self.V_inv_sqrt_
        return self.logreg_.predict_proba(F_eval)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class QuantumKernelSVCClassifier(BaseEstimator, ClassifierMixin):
    """Quantum Fidelity Kernel SVC (QSVC) with 5-fold out-of-fold Platt probability calibration."""

    def __init__(
        self,
        n_qubits: int = 6,
        reps: int = 1,
        bandwidth: float = 0.6,
        C: float = 2.0,
        projection_mode: str = "select_k",
        random_state: int = MODEL_SEED,
    ) -> None:
        self.n_qubits = n_qubits
        self.reps = reps
        self.bandwidth = bandwidth
        self.C = C
        self.projection_mode = projection_mode
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumKernelSVCClassifier":
        y_arr = np.asarray(y, dtype=int)
        self.classes_ = np.array([0, 1])
        self.compressor_ = QuantumAngleCompressor(
            n_qubits=self.n_qubits, bandwidth=self.bandwidth, mode=self.projection_mode
        )
        self.X_train_angles_ = self.compressor_.fit_transform(X, y_arr)
        self.feature_map_ = _build_zero_centered_feature_map(self.n_qubits, reps=self.reps)
        self.phi_train_ = _compute_feature_state_matrix(self.feature_map_, self.X_train_angles_)
        K_train = np.abs(self.phi_train_ @ self.phi_train_.conj().T) ** 2

        inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        oof_scores = np.zeros(len(y_arr), dtype=float)
        for tr_i, va_i in inner_skf.split(K_train, y_arr):
            inner_svc = SVC(kernel="precomputed", C=self.C, random_state=self.random_state)
            inner_svc.fit(K_train[np.ix_(tr_i, tr_i)], y_arr[tr_i])
            oof_scores[va_i] = inner_svc.decision_function(K_train[np.ix_(va_i, tr_i)])

        self.platt_ = LogisticRegression(random_state=self.random_state)
        self.platt_.fit(oof_scores.reshape(-1, 1), y_arr)

        self.svc_ = SVC(kernel="precomputed", C=self.C, random_state=self.random_state)
        self.svc_.fit(K_train, y_arr)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_angles = self.compressor_.transform(X)
        phi_eval = _compute_feature_state_matrix(self.feature_map_, X_angles)
        K_eval = np.abs(phi_eval @ self.phi_train_.conj().T) ** 2
        scores = self.svc_.decision_function(K_eval).reshape(-1, 1)
        return self.platt_.predict_proba(scores)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class HybridVQCClassifier(BaseEstimator, ClassifierMixin):
    """Hybrid Variational Quantum Classifier (Zero-Centered PauliFeatureMap + RealAmplitudes)."""

    def __init__(
        self,
        n_qubits: int = 6,
        feature_reps: int = 1,
        ansatz_reps: int = 2,
        l2_reg: float = 1e-2,
        maxiter: int = 100,
        bandwidth: float = 0.7,
        projection_mode: str = "select_k",
        random_state: int = MODEL_SEED,
    ) -> None:
        self.n_qubits = n_qubits
        self.feature_reps = feature_reps
        self.ansatz_reps = ansatz_reps
        self.l2_reg = l2_reg
        self.maxiter = maxiter
        self.bandwidth = bandwidth
        self.projection_mode = projection_mode
        self.random_state = random_state

    def _build_circuit(self):
        fm = _build_zero_centered_feature_map(self.n_qubits, reps=self.feature_reps)
        ansatz = real_amplitudes(
            num_qubits=self.n_qubits,
            reps=self.ansatz_reps,
            entanglement="linear",
            parameter_prefix="theta",
        )
        qc = fm.compose(ansatz)
        pauli_terms = []
        for q in range(self.n_qubits):
            chars = ["I"] * self.n_qubits
            chars[self.n_qubits - 1 - q] = "Z"
            pauli_terms.append(("".join(chars), 1.0 / self.n_qubits))
        observable = SparsePauliOp.from_list(pauli_terms)
        return qc, fm, ansatz, list(fm.parameters), list(ansatz.parameters), observable

    def _expectation_from_phi(self, phi_matrix: np.ndarray, theta: np.ndarray) -> np.ndarray:
        u_theta = Operator(self.ansatz_.assign_parameters(theta)).data
        psi = u_theta @ phi_matrix
        probs = np.abs(psi) ** 2
        return self.obs_diag_ @ probs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HybridVQCClassifier":
        y_arr = np.asarray(y, dtype=float)
        self.classes_ = np.array([0, 1])
        self.compressor_ = QuantumAngleCompressor(
            n_qubits=self.n_qubits, bandwidth=self.bandwidth, mode=self.projection_mode
        )
        X_angles = self.compressor_.fit_transform(X, y_arr)
        (
            self.circuit_,
            self.fm_,
            self.ansatz_,
            self.fm_params_,
            self.ansatz_params_,
            self.observable_,
        ) = self._build_circuit()

        states = np.arange(1 << self.n_qubits, dtype=np.uint32)
        diag = np.zeros(1 << self.n_qubits, dtype=float)
        for q in range(self.n_qubits):
            diag += 1.0 - 2.0 * ((states >> q) & 1).astype(float)
        self.obs_diag_ = diag / self.n_qubits

        phi_train = _compute_feature_state_matrix(self.fm_, X_angles).T

        rng = np.random.default_rng(self.random_state)
        n_theta = len(self.ansatz_params_)
        init_theta = rng.uniform(-0.3, 0.3, size=n_theta)
        base_rate = float(np.clip(np.mean(y_arr), 1e-3, 1.0 - 1e-3))
        init_bias = float(np.log(base_rate / (1.0 - base_rate)))
        x0 = np.concatenate([init_theta, [2.0, init_bias]])

        def loss_fn(params: np.ndarray) -> float:
            theta = params[:-2]
            scale_w, bias = params[-2], params[-1]
            exp_z = self._expectation_from_phi(phi_train, theta)
            logits = np.clip(scale_w * exp_z + bias, -20.0, 20.0)
            probs = expit(logits)
            bce = -np.mean(
                y_arr * np.log(np.clip(probs, 1e-7, 1.0))
                + (1.0 - y_arr) * np.log(np.clip(1.0 - probs, 1e-7, 1.0))
            )
            reg = self.l2_reg * float(np.sum(theta**2))
            return float(bce + reg)

        res = minimize(
            loss_fn,
            x0,
            method="COBYLA",
            options={"maxiter": self.maxiter, "rhobeg": 0.4},
        )
        self.opt_params_ = res.x
        self.theta_ = res.x[:-2]
        self.scale_w_ = float(res.x[-2])
        self.bias_ = float(res.x[-1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_angles = self.compressor_.transform(X)
        phi_eval = _compute_feature_state_matrix(self.fm_, X_angles).T
        exp_z = self._expectation_from_phi(phi_eval, self.theta_)
        p1 = expit(np.clip(self.scale_w_ * exp_z + self.bias_, -20.0, 20.0))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_candidate_models() -> Dict[str, Any]:
    """Return dictionary of classical baselines and quantum/hybrid classifiers."""
    return {
        "Classical_LogReg": LogisticRegression(
            C=0.25, max_iter=2000, random_state=MODEL_SEED
        ),
        "Classical_RF": RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=5,
            random_state=MODEL_SEED,
        ),
        "Classical_RBF_SVC": CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, random_state=MODEL_SEED),
            cv=5,
            ensemble=False,
        ),
        "Quantum_Kernel_LogReg": QuantumKernelLogRegClassifier(
            n_qubits=6, reps=1, bandwidth=0.7, C=2.0, projection_mode="select_k", random_state=MODEL_SEED
        ),
        "Quantum_Kernel_SVC": QuantumKernelSVCClassifier(
            n_qubits=6, reps=1, bandwidth=0.6, C=2.0, projection_mode="select_k", random_state=MODEL_SEED
        ),
        "Hybrid_VQC": HybridVQCClassifier(
            n_qubits=6, feature_reps=1, ansatz_reps=2, maxiter=100, bandwidth=0.7, projection_mode="select_k", random_state=MODEL_SEED
        ),
    }


def _build_realistic_noise_model() -> NoiseModel:
    """Construct realistic depolarizing + readout noise model for Aer verification."""
    noise_model = NoiseModel()
    error_1q = depolarizing_error(0.001, 1)
    error_2q = depolarizing_error(0.01, 2)
    noise_model.add_all_qubit_quantum_error(error_1q, ["rx", "ry", "rz", "sx", "x", "h", "p", "u"])
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx", "ecr", "cz"])
    readout = ReadoutError([[0.985, 0.015], [0.025, 0.975]])
    noise_model.add_all_qubit_readout_error(readout)
    return noise_model


def verify_quantum_kernel_aer_sampling(
    model: QuantumKernelLogRegClassifier,
    X_sample_raw: pd.DataFrame,
    preprocessor: Pipeline,
    shots: int = 2048,
) -> Dict[str, Any]:
    """Compare ideal statevector quantum kernel vs finite-shot & noisy Aer ComputeUncompute kernel."""
    X_proc = preprocessor.transform(X_sample_raw)
    angles = model.compressor_.transform(X_proc)
    fm = model.feature_map_

    phi_sub = _compute_feature_state_matrix(fm, angles)
    K_ideal = np.abs(phi_sub @ phi_sub.conj().T) ** 2

    pm_ideal = generate_preset_pass_manager(
        optimization_level=1, backend=AerSimulator(), seed_transpiler=SIMULATOR_SEED
    )
    aer_sampler_ideal = AerSamplerV2(default_shots=shots, seed=SIMULATOR_SEED)
    fidelity_shots = ComputeUncompute(sampler=aer_sampler_ideal, pass_manager=pm_ideal)
    qk_shots = FidelityQuantumKernel(feature_map=fm, fidelity=fidelity_shots)
    K_shots = qk_shots.evaluate(angles)

    noise_model = _build_realistic_noise_model()
    noisy_backend = AerSimulator(noise_model=noise_model)
    pm_noisy = generate_preset_pass_manager(
        optimization_level=1, backend=noisy_backend, seed_transpiler=SIMULATOR_SEED
    )
    aer_sampler_noisy = AerSamplerV2(
        default_shots=shots,
        seed=SIMULATOR_SEED,
        options={"backend_options": {"noise_model": noise_model}},
    )
    fidelity_noisy = ComputeUncompute(sampler=aer_sampler_noisy, pass_manager=pm_noisy)
    qk_noisy = FidelityQuantumKernel(feature_map=fm, fidelity=fidelity_noisy)
    K_noisy = qk_noisy.evaluate(angles)

    qc_bound = fm.assign_parameters(angles[0])
    t_qc = transpile(qc_bound, backend=AerSimulator(), optimization_level=1, seed_transpiler=SIMULATOR_SEED)
    circ_metrics = extract_circuit_metrics(qc_bound, t_qc)

    return {
        "subsample_size": int(len(angles)),
        "shots": int(shots),
        "kernel_mae_shots_vs_ideal": float(np.mean(np.abs(K_shots - K_ideal))),
        "kernel_mae_noisy_vs_ideal": float(np.mean(np.abs(K_noisy - K_ideal))),
        "feature_map_circuit_metrics": circ_metrics,
    }


def run_stage1(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    primary_quantum_model_name: str = "Quantum_Kernel_LogReg",
) -> Dict[str, Any]:
    """Run leakage-free 5-fold Stratified CV on train_df, fit final models, and predict test_df once."""
    y_train = train_df["y_binary"].to_numpy(dtype=int)
    skf = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=CV_SPLIT_SEED)

    models = build_candidate_models()
    oof_preds: Dict[str, np.ndarray] = {name: np.zeros(len(train_df), dtype=float) for name in models}
    fold_metrics: Dict[str, List[Dict[str, float]]] = {name: [] for name in models}

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train_df, y_train)):
        df_tr = train_df.iloc[tr_idx]
        df_va = train_df.iloc[va_idx]
        y_tr = y_train[tr_idx]
        y_va = y_train[va_idx]

        prep = build_tabular_preprocessor()
        X_tr = prep.fit_transform(df_tr, y_tr)
        X_va = prep.transform(df_va)

        for name, template_model in models.items():
            clf = clone(template_model)
            clf.fit(X_tr, y_tr)
            p_va = clf.predict_proba(X_va)[:, 1]
            oof_preds[name][va_idx] = p_va
            m = compute_stage1_metrics(y_va, p_va, k=TOP_K_CANDIDATES)
            m["fold"] = fold_idx
            fold_metrics[name].append(m)

    cv_summary: Dict[str, Dict[str, Any]] = {}
    for name in models:
        oof_m = compute_stage1_metrics(y_train, oof_preds[name], k=TOP_K_CANDIDATES)
        fold_df = pd.DataFrame(fold_metrics[name])
        cv_summary[name] = {
            "oof_metrics": oof_m,
            "mean_fold_metrics": {
                col: float(fold_df[col].mean())
                for col in fold_df.columns
                if col != "fold"
            },
            "std_fold_metrics": {
                col: float(fold_df[col].std(ddof=1))
                for col in fold_df.columns
                if col != "fold"
            },
        }

    final_prep = build_tabular_preprocessor()
    X_train_full = final_prep.fit_transform(train_df, y_train)
    X_test_full = final_prep.transform(test_df)

    fitted_models: Dict[str, Any] = {}
    test_probs_by_model: Dict[str, np.ndarray] = {}
    for name, template_model in models.items():
        clf = clone(template_model)
        clf.fit(X_train_full, y_train)
        fitted_models[name] = clf
        test_probs_by_model[name] = np.clip(clf.predict_proba(X_test_full)[:, 1], 0.0, 1.0)

    primary_probs = test_probs_by_model[primary_quantum_model_name]

    aer_kernel_diag = verify_quantum_kernel_aer_sampling(
        fitted_models["Quantum_Kernel_LogReg"],
        train_df.iloc[:12],
        final_prep,
        shots=2048,
    )

    vqc_model: HybridVQCClassifier = fitted_models["Hybrid_VQC"]
    vqc_sample_angles = vqc_model.compressor_.transform(X_test_full[:1])[0]
    vqc_bound = vqc_model.circuit_.assign_parameters(
        {
            **dict(zip(vqc_model.fm_params_, vqc_sample_angles)),
            **dict(zip(vqc_model.ansatz_params_, vqc_model.theta_)),
        }
    )
    vqc_transpiled = transpile(
        vqc_bound, backend=AerSimulator(), optimization_level=1, seed_transpiler=SIMULATOR_SEED
    )
    vqc_circ_metrics = extract_circuit_metrics(vqc_bound, vqc_transpiled)

    result_df = pd.DataFrame(
        {
            "customer_id": test_df["customer_id"].astype(str),
            "predicted_probability": primary_probs,
            "classical_logreg_probability": test_probs_by_model["Classical_LogReg"],
            "classical_rf_probability": test_probs_by_model["Classical_RF"],
            "classical_rbf_svc_probability": test_probs_by_model["Classical_RBF_SVC"],
            "quantum_kernel_svc_probability": test_probs_by_model["Quantum_Kernel_SVC"],
            "hybrid_vqc_probability": test_probs_by_model["Hybrid_VQC"],
        }
    )
    result_df = result_df.sort_values(
        by=["predicted_probability", "customer_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result_df["rank"] = np.arange(1, len(result_df) + 1)
    result_df["top_10_flag"] = (result_df["rank"] <= TOP_K_CANDIDATES).astype(int)

    ordered_cols = [
        "customer_id",
        "predicted_probability",
        "rank",
        "top_10_flag",
        "classical_logreg_probability",
        "classical_rf_probability",
        "classical_rbf_svc_probability",
        "quantum_kernel_svc_probability",
        "hybrid_vqc_probability",
    ]
    result_df = result_df[ordered_cols]

    return {
        "primary_model_name": primary_quantum_model_name,
        "result_df": result_df,
        "oof_predictions": {k: v.tolist() for k, v in oof_preds.items()},
        "y_train": y_train.tolist(),
        "cv_summary": cv_summary,
        "aer_kernel_diagnostics": aer_kernel_diag,
        "vqc_circuit_metrics": vqc_circ_metrics,
    }
