"""Configuration constants, paths, constraints, and reproducibility seeds."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_XLSX_PATH = PROJECT_ROOT / "bank_campaign_participant_release.xlsx"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Top-level deliverable files required by the challenge specification
STAGE1_RESULT_CSV = PROJECT_ROOT / "stage1_results.csv"
STAGE2_RESULT_CSV = PROJECT_ROOT / "stage2_results.csv"
STAGE3_RESULT_CSV = PROJECT_ROOT / "stage3_results.csv"

# Synced root-level alias filenames requested by Devpost submission checks
STAGE1_PREDICTIONS_CSV = PROJECT_ROOT / "stage1_predictions.csv"
STAGE2_SELECTION_CSV = PROJECT_ROOT / "stage2_selection.csv"
STAGE3_SIMULATION_CSV = PROJECT_ROOT / "stage3_simulation.csv"

# Reproducibility seeds (all recorded in pipeline_metadata.json)
GLOBAL_SEED = 42
CV_SPLIT_SEED = 42
MODEL_SEED = 42
OPTIMIZER_SEED = 42
SIMULATOR_SEED = 42
TRANSPILER_SEED = 42
MONTE_CARLO_SEED = 42

# Stage 1 settings
CV_N_SPLITS = 5
TOP_K_CANDIDATES = 10

# Stage 2 constraints (verified against Stage2_Constraints sheet)
STAGE2_EXACT_K = 6
STAGE2_BUDGET_C = 25
STAGE2_CAPACITY_H = 22

# Stage 3 target
STAGE3_MIN_SUBSCRIBERS = 3
STAGE3_SHOTS = 8192
MONTE_CARLO_SHOTS = 100000
STAGE2_QAOA_SHOTS = 4096

