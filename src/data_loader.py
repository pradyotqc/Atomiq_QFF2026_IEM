"""Dataset loading and schema validation for bank_campaign_participant_release.xlsx."""

from pathlib import Path
from typing import Dict
import pandas as pd

from .config import DATA_XLSX_PATH


def load_campaign_data(xlsx_path: Path = DATA_XLSX_PATH) -> Dict[str, pd.DataFrame]:
    """Load and validate all workbook sheets into clean DataFrames."""
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Input workbook not found: {xlsx_path}")

    readme_df = pd.read_excel(xlsx_path, sheet_name="README")
    train_df = pd.read_excel(xlsx_path, sheet_name="Stage1_Train")
    test_df = pd.read_excel(xlsx_path, sheet_name="Stage1_Test")
    stage2_data_df = pd.read_excel(xlsx_path, sheet_name="Stage2_Data")
    stage2_constraints_df = pd.read_excel(xlsx_path, sheet_name="Stage2_Constraints")

    # Enforce strict schema checks
    if "duration" in train_df.columns or "duration" in test_df.columns:
        raise ValueError("Unexpected 'duration' leak column found in Stage1 data.")
    if len(train_df) != 300 or len(test_df) != 100:
        raise ValueError(
            f"Unexpected row counts: train={len(train_df)}, test={len(test_df)}"
        )
    if train_df["customer_id"].duplicated().any() or test_df["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer_id detected in Stage 1 sheets.")

    missing_in_stage2 = set(test_df["customer_id"]) - set(stage2_data_df["customer_id"])
    if missing_in_stage2:
        raise ValueError(f"Test customer IDs missing from Stage2_Data: {missing_in_stage2}")

    train_df["y_binary"] = (train_df["y"].astype(str).str.strip().str.lower() == "yes").astype(int)

    return {
        "readme": readme_df,
        "train": train_df,
        "test": test_df,
        "stage2_data": stage2_data_df,
        "stage2_constraints": stage2_constraints_df,
    }

