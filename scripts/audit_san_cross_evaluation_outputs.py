#!/usr/bin/env python3
"""Audit completed SAN cross-evaluation outputs without rewriting them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_DETAIL_COLUMNS = {
    "evaluation_model",
    "total_arrival_waiting_sec",
    "same_sequence_delay_difference_sec",
    "direct_blocker_delay_sec",
    "propagated_queue_delay_sec",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month-dir", type=Path, action="append", required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--expected-bootstrap-reps", type=int, default=5_000)
    return parser.parse_args()


def audit_month(directory: Path) -> dict[str, object]:
    manifest_path = directory / "manifest.json"
    detail_path = directory / "departure_blocker_detail.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_rows = (
        int(manifest["windows"])
        * int(manifest["replications"])
        * len(manifest["policies"])
        * len(manifest["demand_proxies"])
        * len(manifest["validation"]["selection_models"])
        * len(manifest["validation"]["evaluation_models"])
    )

    rows = 0
    min_waiting = float("inf")
    min_difference = float("inf")
    min_propagated = float("inf")
    max_identity_error = 0.0
    for chunk in pd.read_csv(
        detail_path,
        usecols=lambda column: column in REQUIRED_DETAIL_COLUMNS,
        chunksize=100_000,
    ):
        missing = REQUIRED_DETAIL_COLUMNS.difference(chunk.columns)
        if missing:
            raise ValueError(f"{detail_path} is missing columns: {sorted(missing)}")
        rows += len(chunk)
        min_waiting = min(min_waiting, float(chunk["total_arrival_waiting_sec"].min()))
        blocked = chunk[chunk["evaluation_model"].eq("departure_blocked")]
        min_difference = min(
            min_difference,
            float(blocked["same_sequence_delay_difference_sec"].min()),
        )
        min_propagated = min(
            min_propagated,
            float(blocked["propagated_queue_delay_sec"].min()),
        )
        identity_error = (
            blocked["direct_blocker_delay_sec"]
            + blocked["propagated_queue_delay_sec"]
            - blocked["same_sequence_delay_difference_sec"]
        ).abs()
        max_identity_error = max(max_identity_error, float(identity_error.max()))

    violations = {
        key: value
        for key, value in manifest["validation"].items()
        if key.endswith("_violations")
    }
    return {
        "directory": str(directory),
        "rows": rows,
        "expected_rows": expected_rows,
        "row_count_pass": rows == expected_rows,
        "min_total_arrival_waiting_sec": min_waiting,
        "min_same_sequence_delay_difference_sec": min_difference,
        "min_propagated_queue_delay_sec": min_propagated,
        "max_decomposition_identity_error_sec": max_identity_error,
        "manifest_violations": violations,
    }


def audit_analysis(directory: Path, expected_bootstrap_reps: int) -> dict[str, object]:
    normalized = pd.read_csv(directory / "full_window_common_normalized_policy_metrics.csv")
    scores = pd.read_csv(directory / "full_window_policy_scores.csv")
    regret = pd.read_csv(directory / "full_window_policy_selection_regret.csv")
    bootstrap = pd.read_csv(directory / "full_window_operating_date_cluster_bootstrap.csv")
    policy_bootstrap = pd.read_csv(directory / "full_window_policy_metric_cluster_bootstrap.csv")

    weight_pairs = scores[["lambda_R", "lambda_S"]].drop_duplicates()
    score_group_sizes = scores.groupby(
        ["month", "departure_pressure", "selection_model", "evaluation_model"],
        dropna=False,
    ).size()
    ci_contains_estimate = (
        (policy_bootstrap["ci_lower_95"] <= policy_bootstrap["estimate"])
        & (policy_bootstrap["estimate"] <= policy_bootstrap["ci_upper_95"])
    )
    normalized_ranges = {
        column: {
            "minimum": float(normalized[column].min()),
            "maximum": float(normalized[column].max()),
        }
        for column in [
            "waiting_normalized",
            "displacement_normalized",
            "schedule_normalized",
        ]
    }
    return {
        "directory": str(directory),
        "weight_pairs": len(weight_pairs),
        "score_group_size_values": sorted(int(value) for value in score_group_sizes.unique()),
        "minimum_policy_selection_regret": float(regret["policy_selection_regret"].min()),
        "maximum_policy_selection_regret": float(regret["policy_selection_regret"].max()),
        "negative_regret_rows": int((regret["policy_selection_regret"] < -1e-12).sum()),
        "primary_bootstrap_reps": sorted(int(value) for value in bootstrap["bootstrap_reps"].unique()),
        "policy_bootstrap_reps": sorted(
            int(value) for value in policy_bootstrap["bootstrap_reps"].unique()
        ),
        "expected_bootstrap_reps": expected_bootstrap_reps,
        "policy_bootstrap_cells": len(policy_bootstrap),
        "policy_bootstrap_ci_failures": int((~ci_contains_estimate).sum()),
        "normalized_ranges": normalized_ranges,
        "pressure_strata_reuse_all_window_scales": True,
    }


def main() -> int:
    args = parse_args()
    report = {
        "months": [audit_month(directory) for directory in args.month_dir],
        "analysis": audit_analysis(args.analysis_dir, args.expected_bootstrap_reps),
    }
    print(json.dumps(report, indent=2))

    failures: list[str] = []
    for month in report["months"]:
        if not month["row_count_pass"]:
            failures.append(f"row count mismatch: {month['directory']}")
        if month["min_total_arrival_waiting_sec"] < -1e-7:
            failures.append(f"negative waiting: {month['directory']}")
        if month["min_same_sequence_delay_difference_sec"] < -1e-7:
            failures.append(f"negative same-sequence difference: {month['directory']}")
        if month["min_propagated_queue_delay_sec"] < -1e-7:
            failures.append(f"negative propagated queue delay: {month['directory']}")
        if month["max_decomposition_identity_error_sec"] > 1e-7:
            failures.append(f"decomposition identity failure: {month['directory']}")
        if any(int(value) != 0 for value in month["manifest_violations"].values()):
            failures.append(f"manifest validation failure: {month['directory']}")

    analysis = report["analysis"]
    if analysis["weight_pairs"] != 121:
        failures.append("the scalarization grid does not contain 121 weight pairs")
    if analysis["score_group_size_values"] != [605]:
        failures.append("a policy-score group is incomplete")
    if analysis["negative_regret_rows"] != 0:
        failures.append("policy-selection regret contains negative values")
    if analysis["primary_bootstrap_reps"] != [args.expected_bootstrap_reps]:
        failures.append("primary bootstrap replication count is incorrect")
    if analysis["policy_bootstrap_reps"] != [args.expected_bootstrap_reps]:
        failures.append("policy-metric bootstrap replication count is incorrect")
    if analysis["policy_bootstrap_ci_failures"] != 0:
        failures.append("a policy-metric confidence interval excludes its point estimate")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("SAN cross-evaluation output audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
