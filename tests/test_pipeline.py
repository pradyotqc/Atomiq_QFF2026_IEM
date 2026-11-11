"""End-to-end verification tests for schema, leakage-free behavior, constraints, and quantum consistency."""

import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    RESULTS_DIR,
    STAGE1_PREDICTIONS_CSV,
    STAGE1_RESULT_CSV,
    STAGE2_BUDGET_C,
    STAGE2_CAPACITY_H,
    STAGE2_EXACT_K,
    STAGE2_RESULT_CSV,
    STAGE2_SELECTION_CSV,
    STAGE3_RESULT_CSV,
    STAGE3_SIMULATION_CSV,
)
from src.data_loader import load_campaign_data


class PipelineVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_campaign_data()
        cls.s1 = pd.read_csv(STAGE1_RESULT_CSV)
        cls.s2 = pd.read_csv(STAGE2_RESULT_CSV)
        cls.s3 = pd.read_csv(STAGE3_RESULT_CSV)
        with open(RESULTS_DIR / "stage2_summary.json", encoding="utf-8") as f:
            cls.s2_json = json.load(f)
        with open(RESULTS_DIR / "stage3_summary.json", encoding="utf-8") as f:
            cls.s3_json = json.load(f)

    def test_stage1_schema_and_ranking(self) -> None:
        self.assertEqual(len(self.s1), 100)
        self.assertEqual(set(self.s1["customer_id"]), set(self.data["test"]["customer_id"]))
        self.assertTrue((self.s1["predicted_probability"] >= 0.0).all())
        self.assertTrue((self.s1["predicted_probability"] <= 1.0).all())
        self.assertEqual(int(self.s1["top_10_flag"].sum()), 10)
        self.assertEqual(list(self.s1["rank"]), list(range(1, 101)))
        # Verify monotonic non-increasing probability with ascending customer_id tie-break
        for i in range(len(self.s1) - 1):
            p1, id1 = self.s1.loc[i, "predicted_probability"], self.s1.loc[i, "customer_id"]
            p2, id2 = self.s1.loc[i + 1, "predicted_probability"], self.s1.loc[i + 1, "customer_id"]
            self.assertTrue(p1 > p2 or (np.isclose(p1, p2) and id1 < id2))

    def test_stage2_constraints_and_traceability(self) -> None:
        top10_ids = set(self.s1.loc[self.s1["top_10_flag"] == 1, "customer_id"])
        self.assertEqual(len(self.s2), 10)
        self.assertEqual(set(self.s2["customer_id"]), top10_ids)

        # Traceability: Stage 2 probabilities match Stage 1 probabilities exactly
        merged = self.s2.merge(self.s1[["customer_id", "predicted_probability"]], on="customer_id", suffixes=("_s2", "_s1"))
        np.testing.assert_allclose(merged["predicted_probability_s2"], merged["predicted_probability_s1"], atol=1e-12)

        # Exact feasibility checks
        selected = self.s2[self.s2["selected_flag"] == 1]
        self.assertEqual(len(selected), STAGE2_EXACT_K)
        self.assertLessEqual(float(selected["contact_cost"].sum()), STAGE2_BUDGET_C + 1e-9)
        self.assertLessEqual(float(selected["capacity_units"].sum()), STAGE2_CAPACITY_H + 1e-9)

        # Verify QUBO ground state matches exact brute-force solution
        gs = self.s2_json["qubo_solvers"]["qubo_details"]["ground_state_verification"]
        self.assertTrue(gs["ground_state_matches_exact"])

    def test_stage3_circuit_and_exact_consistency(self) -> None:
        selected_s2 = self.s2[self.s2["selected_flag"] == 1].sort_values(["rank", "customer_id"])
        self.assertEqual(len(self.s3), STAGE2_EXACT_K)
        self.assertEqual(list(self.s3["customer_id"]), list(selected_s2["customer_id"]))

        # Traceability: Stage 3 probabilities match Stage 1/2 probabilities exactly
        np.testing.assert_allclose(
            self.s3["predicted_probability"].to_numpy(),
            selected_s2["predicted_probability"].to_numpy(),
            atol=1e-12,
        )

        # Verify Ry rotation angles satisfy sin^2(theta/2) = p
        p_from_theta = np.sin(self.s3["ry_theta_rad"].to_numpy() / 2.0) ** 2
        np.testing.assert_allclose(p_from_theta, self.s3["predicted_probability"].to_numpy(), atol=1e-10)

        # Verify explicit Stage 3 measurement count columns match shots
        self.assertIn("marginal_ones_count_shots", self.s3.columns)
        self.assertIn("quantum_success_count_shots", self.s3.columns)
        self.assertTrue(
            (self.s3["marginal_ones_count_shots"] + self.s3["marginal_zeros_count_shots"] == self.s3["shots"]).all()
        )
        hw_counts = json.loads(self.s3["hamming_weight_counts_shots_json"].iloc[0])
        self.assertEqual(sum(hw_counts.values()), int(self.s3["shots"].iloc[0]))

        # Ideal statevector must match exact Poisson-Binomial to floating-point precision
        p_sv = self.s3_json["quantum_results"]["ideal_statevector"]["probability_at_least_3"]
        p_exact = self.s3_json["exact_poisson_binomial"]["exact_probability_at_least_k"]
        self.assertAlmostEqual(p_sv, p_exact, places=10)

        # Monte Carlo (100k) should match exact within ~5 standard errors (< 0.01)
        p_mc = self.s3_json["monte_carlo_100k"]["probability_at_least_k"]
        self.assertLess(abs(p_mc - p_exact), 0.01)

    def test_devpost_csv_aliases_in_sync(self) -> None:
        pairs = [
            (STAGE1_RESULT_CSV, STAGE1_PREDICTIONS_CSV),
            (STAGE2_RESULT_CSV, STAGE2_SELECTION_CSV),
            (STAGE3_RESULT_CSV, STAGE3_SIMULATION_CSV),
        ]
        for primary, alias in pairs:
            self.assertTrue(alias.exists(), f"Missing alias file: {alias}")
            self.assertEqual(primary.read_bytes(), alias.read_bytes())


if __name__ == "__main__":
    unittest.main()

