#!/usr/bin/env python3
"""Analyze AO-to-DB versus DB-to-DB selection-model effects at SAN."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("arrival_only", "departure_blocked")
MODEL_INDEPENDENT_POLICIES = {"ObservedRunwayOrder", "DemandFCFS"}
POLICY_DISPLAY_NAMES = {
    "ObservedRunwayOrder": "Observed Runway Order",
    "DemandFCFS": "Demand FCFS",
    "CompletionGreedy": "Completion Greedy",
    "CPSDelay_m2K8": "CPS Delay",
    "ScheduleAware_m2K8": "Schedule Aware",
}
DEFAULT_BOOTSTRAP_REPS = 5_000
DEFAULT_SEED = 8_352_026


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feb-detail",
        type=Path,
        default=ROOT
        / "outputs/san_multimonth/2026_02/full_window_cross_evaluation_r30"
        / "departure_blocker_detail.csv",
    )
    parser.add_argument(
        "--mar-detail",
        type=Path,
        default=ROOT
        / "outputs/san_multimonth/2026_03/full_window_cross_evaluation_r30"
        / "departure_blocker_detail.csv",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=ROOT
        / "outputs/san_multimonth/full_window_cross_evaluation_r30_analysis"
        / "full_window_policy_scores.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/san_multimonth/full_window_cross_evaluation_r30_off_diagonal_analysis",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def read_paired_selection_results(month: str, path: Path) -> pd.DataFrame:
    usecols = [
        "window_id",
        "operating_date_local",
        "replication",
        "policy",
        "selection_model",
        "evaluation_model",
        "sequence_order",
        "total_arrival_waiting_sec",
        "sequence_displacement",
        "absolute_schedule_deviation_sec",
    ]
    detail = pd.read_csv(path, usecols=usecols)
    detail = detail[detail["evaluation_model"].eq("departure_blocked")].copy()
    keys = ["window_id", "operating_date_local", "replication", "policy"]

    duplicate_counts = detail.groupby([*keys, "selection_model"], dropna=False).size()
    if int(duplicate_counts.max()) != 1:
        raise ValueError(f"{month} contains duplicate selection-model rows.")
    selection_counts = detail.groupby(keys, dropna=False)["selection_model"].nunique()
    if not bool((selection_counts == len(MODEL_NAMES)).all()):
        raise ValueError(f"{month} contains incomplete AO-to-DB or DB-to-DB pairs.")

    arrival_only = detail[detail["selection_model"].eq("arrival_only")].drop(
        columns=["selection_model", "evaluation_model"]
    )
    departure_blocked = detail[detail["selection_model"].eq("departure_blocked")].drop(
        columns=["selection_model", "evaluation_model"]
    )
    paired = arrival_only.merge(
        departure_blocked,
        on=keys,
        how="inner",
        suffixes=("_ao_to_db", "_db_to_db"),
        validate="one_to_one",
    )
    paired["month"] = month
    paired["sequence_changed"] = paired["sequence_order_ao_to_db"].ne(
        paired["sequence_order_db_to_db"]
    )
    paired["waiting_difference_sec"] = (
        paired["total_arrival_waiting_sec_ao_to_db"]
        - paired["total_arrival_waiting_sec_db_to_db"]
    )
    paired["displacement_difference"] = (
        paired["sequence_displacement_ao_to_db"]
        - paired["sequence_displacement_db_to_db"]
    )
    paired["schedule_deviation_difference_sec"] = (
        paired["absolute_schedule_deviation_sec_ao_to_db"]
        - paired["absolute_schedule_deviation_sec_db_to_db"]
    )
    return paired


def summarize_policy_effects(paired: pd.DataFrame) -> pd.DataFrame:
    result = (
        paired.groupby(["month", "policy"], dropna=False)
        .agg(
            window_replication_pairs=("sequence_changed", "size"),
            windows=("window_id", "nunique"),
            operating_dates=("operating_date_local", "nunique"),
            sequence_difference_share=("sequence_changed", "mean"),
            mean_waiting_difference_sec=("waiting_difference_sec", "mean"),
            positive_waiting_difference_share=(
                "waiting_difference_sec",
                lambda values: float((values > 1e-9).mean()),
            ),
            negative_waiting_difference_share=(
                "waiting_difference_sec",
                lambda values: float((values < -1e-9).mean()),
            ),
            mean_displacement_difference=("displacement_difference", "mean"),
            mean_schedule_deviation_difference_sec=(
                "schedule_deviation_difference_sec",
                "mean",
            ),
        )
        .reset_index()
    )
    result["policy_display_name"] = result["policy"].map(POLICY_DISPLAY_NAMES)
    independent = result[result["policy"].isin(MODEL_INDEPENDENT_POLICIES)]
    if not bool((independent["sequence_difference_share"] == 0.0).all()):
        raise ValueError("A model-independent policy changed sequence across selection models.")
    if not bool(np.isclose(independent["mean_waiting_difference_sec"], 0.0).all()):
        raise ValueError("A model-independent policy changed DB-evaluated waiting.")
    return result.sort_values(["month", "policy"], kind="mergesort")


def cluster_bootstrap_completion_greedy(
    paired: pd.DataFrame,
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    completion_greedy = paired[paired["policy"].eq("CompletionGreedy")].copy()
    window_level = (
        completion_greedy.groupby(
            ["month", "operating_date_local", "window_id"],
            dropna=False,
        )
        .agg(
            waiting_difference_sec=("waiting_difference_sec", "mean"),
            sequence_difference_share=("sequence_changed", "mean"),
        )
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for month, chunk in window_level.groupby("month", sort=True):
        chunk = chunk.reset_index(drop=True)
        dates = sorted(chunk["operating_date_local"].dropna().unique().tolist())
        date_index = {date: index for index, date in enumerate(dates)}
        row_date_index = chunk["operating_date_local"].map(date_index).to_numpy(dtype=int)
        metrics = {
            "waiting_difference_sec_per_window": chunk["waiting_difference_sec"].to_numpy(
                dtype=float
            ),
            "sequence_difference_share": chunk["sequence_difference_share"].to_numpy(
                dtype=float
            ),
        }
        draws = {
            metric: np.empty(bootstrap_reps, dtype=float) for metric in metrics
        }
        for draw in range(bootstrap_reps):
            sampled_dates = rng.integers(0, len(dates), size=len(dates))
            date_multiplicity = np.bincount(sampled_dates, minlength=len(dates))
            weights = date_multiplicity[row_date_index].astype(float)
            denominator = float(weights.sum())
            for metric, values in metrics.items():
                draws[metric][draw] = float(np.sum(weights * values) / denominator)
        for metric, values in metrics.items():
            rows.append(
                {
                    "month": month,
                    "policy": "CompletionGreedy",
                    "policy_display_name": POLICY_DISPLAY_NAMES["CompletionGreedy"],
                    "metric": metric,
                    "estimate": float(np.mean(values)),
                    "ci_lower_95": float(np.quantile(draws[metric], 0.025)),
                    "ci_upper_95": float(np.quantile(draws[metric], 0.975)),
                    "operating_date_clusters": len(dates),
                    "bootstrap_reps": bootstrap_reps,
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows)


def summarize_weight_grid(scores_path: Path) -> pd.DataFrame:
    scores = pd.read_csv(scores_path)
    scores = scores[scores["evaluation_model"].eq("departure_blocked")].copy()
    group_cols = ["month", "departure_pressure", "lambda_R", "lambda_S"]
    rows: list[dict[str, object]] = []
    for keys, chunk in scores.groupby(group_cols, dropna=False, sort=True):
        arrival_only = chunk[chunk["selection_model"].eq("arrival_only")]
        departure_blocked = chunk[chunk["selection_model"].eq("departure_blocked")]
        ranks = arrival_only[["policy", "policy_rank"]].merge(
            departure_blocked[["policy", "policy_rank"]],
            on="policy",
            how="inner",
            suffixes=("_ao_to_db", "_db_to_db"),
            validate="one_to_one",
        )
        if len(ranks) != len(POLICY_DISPLAY_NAMES):
            raise ValueError(f"Incomplete policy rank comparison for {keys}.")
        preferred_ao = set(arrival_only.loc[arrival_only["is_preferred"], "policy"])
        preferred_db = set(
            departure_blocked.loc[departure_blocked["is_preferred"], "policy"]
        )
        rows.append(
            dict(zip(group_cols, keys))
            | {
                "any_policy_rank_difference": bool(
                    (ranks["policy_rank_ao_to_db"] != ranks["policy_rank_db_to_db"]).any()
                ),
                "preferred_policy_set_difference": preferred_ao != preferred_db,
            }
        )
    comparisons = pd.DataFrame(rows)
    return (
        comparisons.groupby(["month", "departure_pressure"], dropna=False)
        .agg(
            weight_combinations=("any_policy_rank_difference", "size"),
            combinations_with_rank_difference=("any_policy_rank_difference", "sum"),
            rank_difference_share=("any_policy_rank_difference", "mean"),
            combinations_with_preferred_set_difference=(
                "preferred_policy_set_difference",
                "sum",
            ),
            preferred_set_difference_share=(
                "preferred_policy_set_difference",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(["month", "departure_pressure"], kind="mergesort")
    )


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paired = pd.concat(
        [
            read_paired_selection_results("2026_02", args.feb_detail),
            read_paired_selection_results("2026_03", args.mar_detail),
        ],
        ignore_index=True,
    )
    policy_effects = summarize_policy_effects(paired)
    bootstrap = cluster_bootstrap_completion_greedy(
        paired,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
    )
    weight_grid = summarize_weight_grid(args.scores)

    policy_effects.to_csv(
        args.out_dir / "off_diagonal_policy_sequence_effects.csv",
        index=False,
    )
    bootstrap.to_csv(
        args.out_dir / "off_diagonal_completion_greedy_cluster_bootstrap.csv",
        index=False,
    )
    weight_grid.to_csv(
        args.out_dir / "off_diagonal_weight_grid_summary.csv",
        index=False,
    )

    print(policy_effects.to_string(index=False))
    print()
    print(bootstrap.to_string(index=False))
    print()
    print(weight_grid.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
