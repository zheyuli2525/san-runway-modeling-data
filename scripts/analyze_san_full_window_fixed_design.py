#!/usr/bin/env python3
"""Analyze SAN full-window cross-evaluation results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_NAMES = ("arrival_only", "departure_blocked")
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
    parser.add_argument("--feb-dir", type=Path, required=True)
    parser.add_argument("--mar-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--demand-proxy", default="terminal_30nm_entry")
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_month(label: str, directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    detail_path = directory / "departure_blocker_detail.csv"
    features_path = directory / "departure_weather_window_features.csv"
    manifest_path = directory / "manifest.json"
    for path in [detail_path, features_path, manifest_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    detail = pd.read_csv(detail_path)
    features = pd.read_csv(features_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    detail["month"] = label
    features["month"] = label
    return detail, features, manifest


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(
            f"{label} is missing {missing}. Rerun run_san_departure_blocker_experiment.py "
            "with the cross-evaluation implementation before this analysis."
        )


def expected_rows(manifest: dict) -> int:
    policies = len(manifest.get("policies", []))
    proxies = len(manifest.get("demand_proxies", []))
    validation = manifest.get("validation", {})
    selection_models = len(validation.get("selection_models", MODEL_NAMES))
    evaluation_models = len(validation.get("evaluation_models", MODEL_NAMES))
    return (
        int(manifest.get("windows", 0))
        * int(manifest.get("replications", 0))
        * policies
        * proxies
        * selection_models
        * evaluation_models
    )


def validation_table(manifests: dict[str, dict], details: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fields = [
        "negative_wait_violations",
        "same_aircraft_set_violations",
        "cps_shift_violations",
        "same_sequence_monotonicity_violations",
        "empty_block_equivalence_violations",
        "decomposition_identity_violations",
        "model_independent_policy_sequence_violations",
    ]
    for month, manifest in manifests.items():
        validation = manifest.get("validation", {})
        row: dict[str, object] = {
            "month": month,
            "windows": manifest.get("windows"),
            "replications": manifest.get("replications"),
            "expected_rows": expected_rows(manifest),
            "actual_rows": len(details[month]),
            "row_count_pass": len(details[month]) == expected_rows(manifest),
            "weather_coverage": manifest.get("weather_window_coverage_rate"),
            "events_total": manifest.get("events_total"),
            "events_schedule_usable": manifest.get("events_schedule_usable"),
        }
        row.update({field: validation.get(field) for field in fields})
        rows.append(row)
    return pd.DataFrame(rows)


def add_departure_pressure(detail: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    occupancy_col = "blocker_total_sec" if "blocker_total_sec" in features else "departure_occupancy_sec"
    if occupancy_col not in features:
        raise ValueError("Window features require blocker_total_sec or departure_occupancy_sec.")
    keep = [
        column
        for column in [
            "month",
            "window_id",
            occupancy_col,
            "n_departures",
            "blocker_count",
            "weather_available",
            "flight_category",
        ]
        if column in features
    ]
    window_features = features[keep].drop_duplicates(["month", "window_id"]).copy()
    window_features = window_features.rename(columns={occupancy_col: "departure_occupancy_sec"})
    window_features["departure_pressure"] = window_features.groupby("month", group_keys=False)[
        "departure_occupancy_sec"
    ].apply(
        lambda values: pd.qcut(
            values.rank(method="first"),
            q=3,
            labels=["low", "medium", "high"],
        )
    )
    return detail.merge(window_features, on=["month", "window_id"], how="left", validate="many_to_one")


def policy_summary(detail: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["month", "selection_model", "evaluation_model", "policy"]
    summary = (
        detail.groupby(group_cols, dropna=False)
        .agg(
            windows=("window_id", "nunique"),
            operating_dates=("operating_date_local", "nunique"),
            rows=("window_id", "size"),
            mean_n_arrivals=("n_arrivals", "mean"),
            mean_total_arrival_waiting_sec=("total_arrival_waiting_sec", "mean"),
            median_total_arrival_waiting_sec=("total_arrival_waiting_sec", "median"),
            mean_sequence_displacement=("sequence_displacement", "mean"),
            mean_absolute_schedule_deviation_sec=("absolute_schedule_deviation_sec", "mean"),
            mean_direct_blocker_delay_sec=("direct_blocker_delay_sec", "mean"),
            mean_same_sequence_delay_difference_sec=("same_sequence_delay_difference_sec", "mean"),
            mean_propagated_queue_delay_sec=("propagated_queue_delay_sec", "mean"),
            mean_sequence_generation_runtime_sec=("sequence_generation_runtime_sec", "mean"),
        )
        .reset_index()
    )
    summary["policy_display_name"] = summary["policy"].map(POLICY_DISPLAY_NAMES).fillna(summary["policy"])
    return summary.sort_values(
        ["month", "selection_model", "evaluation_model", "mean_total_arrival_waiting_sec", "mean_sequence_displacement"],
        kind="mergesort",
    )


def same_sequence_detail(detail: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "month",
        "window_id",
        "operating_date_local",
        "replication",
        "selection_model",
        "policy",
    ]
    checks = (
        detail.groupby(keys, dropna=False)
        .agg(
            evaluation_models=("evaluation_model", "nunique"),
            sequence_versions=("sequence_order", "nunique"),
        )
        .reset_index()
    )
    if not bool((checks["evaluation_models"] == len(MODEL_NAMES)).all()):
        raise ValueError("At least one generated sequence was not evaluated under both runway models.")
    if not bool((checks["sequence_versions"] == 1).all()):
        raise ValueError("At least one evaluation regenerated or changed a fixed sequence.")

    pair = detail[detail["evaluation_model"].eq("departure_blocked")][
        [
            *keys,
            "n_arrivals",
            "sequence_order",
            "sequence_displacement",
            "same_sequence_delay_difference_sec",
            "direct_blocker_delay_sec",
            "propagated_queue_delay_sec",
            "departure_occupancy_sec",
            "departure_pressure",
        ]
    ].copy()
    pair["propagated_share"] = np.where(
        pair["same_sequence_delay_difference_sec"] > 1e-9,
        pair["propagated_queue_delay_sec"] / pair["same_sequence_delay_difference_sec"],
        np.nan,
    )
    return pair


def same_sequence_summary(pair: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    strata = [("all", pair)] + [
        (name, pair[pair["departure_pressure"].eq(name)])
        for name in ["low", "medium", "high"]
    ]
    for stratum, subset in strata:
        if subset.empty:
            continue
        grouped = (
            subset.groupby(["month", "selection_model", "policy"], dropna=False)
            .agg(
                windows=("window_id", "nunique"),
                operating_dates=("operating_date_local", "nunique"),
                mean_departure_occupancy_sec=("departure_occupancy_sec", "mean"),
                mean_same_sequence_delay_difference_sec=("same_sequence_delay_difference_sec", "mean"),
                mean_direct_blocker_delay_sec=("direct_blocker_delay_sec", "mean"),
                mean_propagated_queue_delay_sec=("propagated_queue_delay_sec", "mean"),
                positive_difference_share=("same_sequence_delay_difference_sec", lambda x: float((x > 1e-9).mean())),
            )
            .reset_index()
        )
        component_total = (
            subset.groupby(["month", "selection_model", "policy"], dropna=False)[
                ["same_sequence_delay_difference_sec", "propagated_queue_delay_sec"]
            ]
            .sum()
            .reset_index()
        )
        component_total["aggregate_propagated_share"] = np.where(
            component_total["same_sequence_delay_difference_sec"] > 1e-9,
            component_total["propagated_queue_delay_sec"]
            / component_total["same_sequence_delay_difference_sec"],
            np.nan,
        )
        grouped = grouped.merge(
            component_total[["month", "selection_model", "policy", "aggregate_propagated_share"]],
            on=["month", "selection_model", "policy"],
            how="left",
            validate="one_to_one",
        )
        grouped["departure_pressure"] = stratum
        frames.append(grouped)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not result.empty:
        result["policy_display_name"] = result["policy"].map(POLICY_DISPLAY_NAMES).fillna(result["policy"])
    return result


def common_complete_keys(subset: pd.DataFrame, policies: list[str], metric: str) -> pd.DataFrame:
    key_cols = ["month", "window_id", "operating_date_local", "replication"]
    expected_cells = len(MODEL_NAMES) * len(MODEL_NAMES) * len(policies)
    eligible = subset[subset[metric].notna()].copy()
    counts = eligible.groupby(key_cols, dropna=False).size().rename("cell_count").reset_index()
    return counts[counts["cell_count"].eq(expected_cells)][key_cols]


def aggregate_policy_metrics_for_stratum(
    subset: pd.DataFrame,
    policies: list[str],
    stratum: str,
) -> pd.DataFrame:
    cell_cols = ["month", "selection_model", "evaluation_model", "policy"]
    metric_specs = [
        ("total_arrival_waiting_sec", "mean_total_arrival_waiting_sec", "waiting_sample_rows"),
        ("sequence_displacement", "mean_sequence_displacement", "displacement_sample_rows"),
        ("absolute_schedule_deviation_sec", "mean_absolute_schedule_deviation_sec", "schedule_sample_rows"),
    ]
    merged: pd.DataFrame | None = None
    for source, target, count_name in metric_specs:
        common_keys = common_complete_keys(subset, policies, source)
        complete = subset.merge(
            common_keys.assign(_common=1),
            on=["month", "window_id", "operating_date_local", "replication"],
            how="inner",
            validate="many_to_one",
        )
        metric_frame = (
            complete.groupby(cell_cols, dropna=False)
            .agg(**{target: (source, "mean"), count_name: (source, "size")})
            .reset_index()
        )
        merged = (
            metric_frame
            if merged is None
            else merged.merge(metric_frame, on=cell_cols, how="outer", validate="one_to_one")
        )
    if merged is None:
        return pd.DataFrame()
    merged["departure_pressure"] = stratum
    return merged


def aggregate_policy_metrics(detail: pd.DataFrame) -> pd.DataFrame:
    policies = sorted(detail["policy"].dropna().unique().tolist())
    frames = [aggregate_policy_metrics_for_stratum(detail, policies, "all")]
    for pressure in ["low", "medium", "high"]:
        frames.append(
            aggregate_policy_metrics_for_stratum(
                detail[detail["departure_pressure"].eq(pressure)].copy(),
                policies,
                pressure,
            )
        )
    metrics = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    metrics["policy_display_name"] = metrics["policy"].map(POLICY_DISPLAY_NAMES).fillna(metrics["policy"])
    return metrics


def apply_common_monthly_normalization(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized = metrics.copy()
    specs = [
        ("mean_total_arrival_waiting_sec", "waiting_normalized"),
        ("mean_sequence_displacement", "displacement_normalized"),
        ("mean_absolute_schedule_deviation_sec", "schedule_normalized"),
    ]
    scale_rows: list[dict[str, object]] = []
    all_window_metrics = metrics[metrics["departure_pressure"].eq("all")]
    for month, all_window_cells in all_window_metrics.groupby("month"):
        for source, target in specs:
            valid = all_window_cells[source].dropna()
            minimum = float(valid.min()) if not valid.empty else np.nan
            maximum = float(valid.max()) if not valid.empty else np.nan
            span = maximum - minimum if np.isfinite(minimum) and np.isfinite(maximum) else np.nan
            scale_rows.append(
                {
                    "month": month,
                    "metric": source,
                    "minimum": minimum,
                    "maximum": maximum,
                    "range": span,
                }
            )
            mask = normalized["month"].eq(month)
            if not np.isfinite(span):
                normalized.loc[mask, target] = np.nan
            elif span <= 1e-12:
                normalized.loc[mask, target] = 0.0
            else:
                normalized.loc[mask, target] = (normalized.loc[mask, source] - minimum) / span
    return normalized, pd.DataFrame(scale_rows)


def score_grid(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (month, pressure), chunk in metrics.groupby(["month", "departure_pressure"], dropna=False):
        for lambda_r in np.linspace(0.0, 1.0, 11):
            for lambda_s in np.linspace(0.0, 1.0, 11):
                scored = chunk.copy()
                if lambda_s > 0 and scored["schedule_normalized"].isna().any():
                    continue
                schedule_term = scored["schedule_normalized"].fillna(0.0)
                scored["policy_score"] = (
                    scored["waiting_normalized"]
                    + lambda_r * scored["displacement_normalized"]
                    + lambda_s * schedule_term
                )
                scored["lambda_R"] = float(lambda_r)
                scored["lambda_S"] = float(lambda_s)
                scored["policy_rank"] = scored.groupby(
                    ["selection_model", "evaluation_model"], dropna=False
                )["policy_score"].rank(method="min", ascending=True)
                minimum = scored.groupby(["selection_model", "evaluation_model"], dropna=False)[
                    "policy_score"
                ].transform("min")
                scored["is_preferred"] = np.isclose(
                    scored["policy_score"], minimum, rtol=0.0, atol=1e-12
                )
                rows.append(scored)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def rank_transitions(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    index = ["month", "departure_pressure", "lambda_R", "lambda_S", "policy", "policy_display_name"]
    wide = scores.pivot_table(
        index=index,
        columns=["selection_model", "evaluation_model"],
        values="policy_rank",
        aggfunc="first",
    )
    wide.columns = [f"rank_{selection}_to_{evaluation}" for selection, evaluation in wide.columns]
    wide = wide.reset_index()
    ao_ao = "rank_arrival_only_to_arrival_only"
    ao_db = "rank_arrival_only_to_departure_blocked"
    db_db = "rank_departure_blocked_to_departure_blocked"
    db_ao = "rank_departure_blocked_to_arrival_only"
    for column in [ao_ao, ao_db, db_db, db_ao]:
        if column not in wide:
            wide[column] = np.nan
    wide["diagonal_rank_transition"] = wide[db_db] - wide[ao_ao]
    wide["evaluation_model_rank_transition"] = wide[ao_db] - wide[ao_ao]
    wide["selection_model_rank_transition"] = wide[db_db] - wide[ao_db]
    return wide


def preferred_policy_and_regret(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    disagreement_rows: list[dict[str, object]] = []
    regret_rows: list[dict[str, object]] = []
    group_cols = ["month", "departure_pressure", "lambda_R", "lambda_S"]
    for keys, chunk in scores.groupby(group_cols, dropna=False):
        month, pressure, lambda_r, lambda_s = keys
        ao_ao = chunk[
            chunk["selection_model"].eq("arrival_only")
            & chunk["evaluation_model"].eq("arrival_only")
        ]
        ao_db = chunk[
            chunk["selection_model"].eq("arrival_only")
            & chunk["evaluation_model"].eq("departure_blocked")
        ]
        db_db = chunk[
            chunk["selection_model"].eq("departure_blocked")
            & chunk["evaluation_model"].eq("departure_blocked")
        ]
        all_db = chunk[chunk["evaluation_model"].eq("departure_blocked")]
        ao_preferred = sorted(ao_ao.loc[ao_ao["is_preferred"], "policy"].tolist())
        db_preferred = sorted(db_db.loc[db_db["is_preferred"], "policy"].tolist())
        all_db_min = float(all_db["policy_score"].min())
        db_diag_min = float(db_db["policy_score"].min())
        all_db_reference = all_db[
            np.isclose(all_db["policy_score"], all_db_min, rtol=0.0, atol=1e-12)
        ]
        reference_sequences = sorted(
            f"{row.selection_model}:{row.policy}" for row in all_db_reference.itertuples(index=False)
        )
        disagreement_rows.append(
            {
                "month": month,
                "departure_pressure": pressure,
                "lambda_R": lambda_r,
                "lambda_S": lambda_s,
                "arrival_only_preferred_policies": " | ".join(ao_preferred),
                "departure_blocked_preferred_policies": " | ".join(db_preferred),
                "preferred_policy_disagreement": int(set(ao_preferred) != set(db_preferred)),
            }
        )
        for policy in ao_preferred:
            selected = ao_db[ao_db["policy"].eq(policy)]
            if selected.empty:
                continue
            selected_score = float(selected.iloc[0]["policy_score"])
            regret_rows.append(
                {
                    "month": month,
                    "departure_pressure": pressure,
                    "lambda_R": lambda_r,
                    "lambda_S": lambda_s,
                    "arrival_only_selected_policy": policy,
                    "arrival_only_selected_policy_display_name": POLICY_DISPLAY_NAMES.get(policy, policy),
                    "arrival_only_selected_sequence_departure_blocked_score": selected_score,
                    "best_cross_evaluated_departure_blocked_score": all_db_min,
                    "best_cross_evaluated_departure_blocked_sequences": " | ".join(reference_sequences),
                    "policy_selection_regret": max(0.0, selected_score - all_db_min),
                    "diagonal_score_difference": selected_score - db_diag_min,
                }
            )
    return pd.DataFrame(disagreement_rows), pd.DataFrame(regret_rows)


def cluster_bootstrap_same_sequence(
    pair: pd.DataFrame,
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    primary = pair[
        pair["selection_model"].eq("arrival_only")
        & pair["policy"].eq("ObservedRunwayOrder")
    ].copy()
    window_level = (
        primary.groupby(
            ["month", "operating_date_local", "window_id", "departure_pressure"],
            observed=True,
            dropna=False,
        )
        .agg(
            same_sequence_delay_difference_sec=("same_sequence_delay_difference_sec", "mean"),
            direct_blocker_delay_sec=("direct_blocker_delay_sec", "mean"),
            propagated_queue_delay_sec=("propagated_queue_delay_sec", "mean"),
        )
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for month, chunk in window_level.groupby("month"):
        chunk = chunk.reset_index(drop=True)
        dates = sorted(chunk["operating_date_local"].dropna().unique().tolist())
        if not dates:
            continue
        date_index = {date: index for index, date in enumerate(dates)}
        row_date_index = chunk["operating_date_local"].map(date_index).to_numpy(dtype=int)
        values = {
            "same_sequence_delay_difference_sec_per_window": chunk[
                "same_sequence_delay_difference_sec"
            ].to_numpy(dtype=float),
            "direct_blocker_delay_sec_per_window": chunk["direct_blocker_delay_sec"].to_numpy(dtype=float),
            "propagated_queue_delay_sec_per_window": chunk[
                "propagated_queue_delay_sec"
            ].to_numpy(dtype=float),
        }
        pressure = chunk["departure_pressure"].astype(str).to_numpy()
        estimates = {name: float(np.mean(array)) for name, array in values.items()}
        total_difference = float(np.sum(values["same_sequence_delay_difference_sec_per_window"]))
        estimates["aggregate_propagated_share"] = (
            float(np.sum(values["propagated_queue_delay_sec_per_window"]) / total_difference)
            if total_difference > 1e-9
            else np.nan
        )
        high_values = values["same_sequence_delay_difference_sec_per_window"][pressure == "high"]
        low_values = values["same_sequence_delay_difference_sec_per_window"][pressure == "low"]
        estimates["high_minus_low_delay_difference_sec_per_window"] = (
            float(np.mean(high_values) - np.mean(low_values))
            if len(high_values) and len(low_values)
            else np.nan
        )
        draws = {name: np.empty(bootstrap_reps, dtype=float) for name in estimates}
        for draw in range(bootstrap_reps):
            sampled_dates = rng.integers(0, len(dates), size=len(dates))
            date_multiplicity = np.bincount(sampled_dates, minlength=len(dates))
            weights = date_multiplicity[row_date_index].astype(float)
            positive = weights > 0
            weight_sum = float(weights.sum())
            for name, array in values.items():
                draws[name][draw] = float(np.sum(weights * array) / weight_sum)
            difference_total = float(
                np.sum(weights * values["same_sequence_delay_difference_sec_per_window"])
            )
            draws["aggregate_propagated_share"][draw] = (
                float(np.sum(weights * values["propagated_queue_delay_sec_per_window"]) / difference_total)
                if difference_total > 1e-9
                else np.nan
            )
            high_mask = positive & (pressure == "high")
            low_mask = positive & (pressure == "low")
            if high_mask.any() and low_mask.any():
                high_mean = float(
                    np.sum(
                        weights[high_mask]
                        * values["same_sequence_delay_difference_sec_per_window"][high_mask]
                    )
                    / np.sum(weights[high_mask])
                )
                low_mean = float(
                    np.sum(
                        weights[low_mask]
                        * values["same_sequence_delay_difference_sec_per_window"][low_mask]
                    )
                    / np.sum(weights[low_mask])
                )
                draws["high_minus_low_delay_difference_sec_per_window"][draw] = high_mean - low_mean
            else:
                draws["high_minus_low_delay_difference_sec_per_window"][draw] = np.nan
        for metric, estimate in estimates.items():
            valid = draws[metric][np.isfinite(draws[metric])]
            rows.append(
                {
                    "month": month,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_lower_95": float(np.quantile(valid, 0.025)) if len(valid) else np.nan,
                    "ci_upper_95": float(np.quantile(valid, 0.975)) if len(valid) else np.nan,
                    "operating_date_clusters": len(dates),
                    "bootstrap_reps": bootstrap_reps,
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows)


def cluster_bootstrap_policy_metrics(
    detail: pd.DataFrame,
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    cell_cols = ["selection_model", "evaluation_model", "policy"]
    metric_columns = [
        "total_arrival_waiting_sec",
        "sequence_displacement",
        "absolute_schedule_deviation_sec",
    ]
    window_level = (
        detail.groupby(
            [
                "month",
                "operating_date_local",
                "window_id",
                "departure_pressure",
                *cell_cols,
            ],
            observed=True,
            dropna=False,
        )[metric_columns]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    cell_index = pd.MultiIndex.from_tuples(
        sorted(
            {
                (row.selection_model, row.evaluation_model, row.policy)
                for row in window_level[cell_cols].itertuples(index=False)
            }
        ),
        names=cell_cols,
    )
    for month, month_frame in window_level.groupby("month"):
        strata = [("all", month_frame)] + [
            (pressure, month_frame[month_frame["departure_pressure"].eq(pressure)])
            for pressure in ["low", "medium", "high"]
        ]
        for pressure, stratum_frame in strata:
            if stratum_frame.empty:
                continue
            dates = sorted(stratum_frame["operating_date_local"].dropna().unique().tolist())
            if not dates:
                continue
            date_lookup = {date: index for index, date in enumerate(dates)}
            for metric in metric_columns:
                pivot = stratum_frame.pivot_table(
                    index=["operating_date_local", "window_id"],
                    columns=cell_cols,
                    values=metric,
                    aggfunc="first",
                ).reindex(columns=cell_index)
                pivot = pivot.dropna(axis=0, how="any")
                if pivot.empty:
                    continue
                values = pivot.to_numpy(dtype=float)
                row_date_index = np.array(
                    [date_lookup[index[0]] for index in pivot.index],
                    dtype=int,
                )
                estimates = np.mean(values, axis=0)
                draws = np.full((bootstrap_reps, len(cell_index)), np.nan, dtype=float)
                for draw in range(bootstrap_reps):
                    sampled_dates = rng.integers(0, len(dates), size=len(dates))
                    date_multiplicity = np.bincount(sampled_dates, minlength=len(dates))
                    weights = date_multiplicity[row_date_index].astype(float)
                    weight_sum = float(weights.sum())
                    if weight_sum > 0:
                        draws[draw, :] = np.sum(values * weights[:, None], axis=0) / weight_sum
                for cell_position, (selection_model, evaluation_model, policy) in enumerate(cell_index):
                    valid = draws[:, cell_position]
                    valid = valid[np.isfinite(valid)]
                    rows.append(
                        {
                            "month": month,
                            "departure_pressure": pressure,
                            "selection_model": selection_model,
                            "evaluation_model": evaluation_model,
                            "policy": policy,
                            "policy_display_name": POLICY_DISPLAY_NAMES.get(policy, policy),
                            "metric": metric,
                            "estimate": float(estimates[cell_position]),
                            "ci_lower_95": float(np.quantile(valid, 0.025)) if len(valid) else np.nan,
                            "ci_upper_95": float(np.quantile(valid, 0.975)) if len(valid) else np.nan,
                            "operating_date_clusters": len(dates),
                            "bootstrap_reps": bootstrap_reps,
                            "seed": seed,
                        }
                    )
    return pd.DataFrame(rows)


def df_to_markdown(df: pd.DataFrame, float_digits: int = 3) -> str:
    if df.empty:
        return "_No rows._"
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.{float_digits}f}"
            )
        else:
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else str(value))
    lines = [
        "| " + " | ".join(out.columns) + " |",
        "| " + " | ".join(["---"] * len(out.columns)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in out.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def write_markdown(
    out_dir: Path,
    validation: pd.DataFrame,
    same_sequence: pd.DataFrame,
    bootstrap: pd.DataFrame,
    policy_bootstrap: pd.DataFrame,
    disagreement: pd.DataFrame,
    regret: pd.DataFrame,
) -> None:
    observed = same_sequence[
        same_sequence["selection_model"].eq("arrival_only")
        & same_sequence["policy"].eq("ObservedRunwayOrder")
        & same_sequence["departure_pressure"].eq("all")
    ]
    disagreement_summary = (
        disagreement.groupby(["month", "departure_pressure"], dropna=False)[
            "preferred_policy_disagreement"
        ]
        .mean()
        .rename("disagreement_share_across_weight_combinations")
        .reset_index()
        if not disagreement.empty
        else pd.DataFrame()
    )
    regret_summary = (
        regret.groupby(["month", "departure_pressure"], dropna=False)["policy_selection_regret"]
        .agg(["mean", "median", "max"])
        .reset_index()
        if not regret.empty
        else pd.DataFrame()
    )
    lines = [
        "# SAN Full-Window Cross-Evaluation Results",
        "",
        f"Generated: {now_iso()}",
        "",
        "## Validation",
        "",
        df_to_markdown(validation),
        "",
        "## Observed Runway Order Same-Sequence Effects",
        "",
        df_to_markdown(observed),
        "",
        "## Operating-Date Cluster Bootstrap",
        "",
        df_to_markdown(bootstrap),
        "",
        "## Preferred-Policy Disagreement",
        "",
        df_to_markdown(disagreement_summary),
        "",
        "## Policy-Selection Regret",
        "",
        df_to_markdown(regret_summary),
        "",
        "## Policy-Metric Bootstrap Coverage",
        "",
        (
            f"Operating-date bootstrap intervals were computed for {len(policy_bootstrap):,} "
            "month-pressure-model-policy-metric cells."
            if not policy_bootstrap.empty
            else "_No rows._"
        ),
        "",
        "## Analysis Notes",
        "",
        "- Every generated sequence is evaluated unchanged under both runway models.",
        "- Monthly min-max scales pool both selection models, both evaluation models, and all five evaluated policies.",
        "- Pressure-stratified means reuse the all-window monthly scales without clipping, so their normalized values are not constrained to [0, 1].",
        "- Simulation replications are averaged within empirical windows before operating dates are resampled.",
        "- Policy-selection regret uses the lowest departure-blocked score among all cross-evaluated sequences as its reference, so the quantity is nonnegative. The separate diagonal score difference retains the originally proposed diagonal comparison.",
        "",
    ]
    (out_dir / "SAN_Full_Window_Cross_Evaluation_Results.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    month_dirs = {"2026_02": args.feb_dir, "2026_03": args.mar_dir}
    details: dict[str, pd.DataFrame] = {}
    features: dict[str, pd.DataFrame] = {}
    manifests: dict[str, dict] = {}
    for month, directory in month_dirs.items():
        month_detail, month_features, manifest = read_month(month, directory)
        details[month] = month_detail
        features[month] = month_features
        manifests[month] = manifest

    required = {
        "window_id",
        "operating_date_local",
        "demand_proxy",
        "replication",
        "policy",
        "selection_model",
        "evaluation_model",
        "sequence_order",
        "total_arrival_waiting_sec",
        "sequence_displacement",
        "absolute_schedule_deviation_sec",
        "direct_blocker_delay_sec",
        "same_sequence_delay_difference_sec",
        "propagated_queue_delay_sec",
    }
    for month, frame in details.items():
        require_columns(frame, required, f"{month} detail")

    all_detail = pd.concat(details.values(), ignore_index=True)
    all_features = pd.concat(features.values(), ignore_index=True)
    all_detail = all_detail[all_detail["demand_proxy"].eq(args.demand_proxy)].copy()
    if all_detail.empty:
        raise ValueError(f"No rows found for demand proxy {args.demand_proxy!r}.")
    all_detail = add_departure_pressure(all_detail, all_features)

    validation = validation_table(manifests, details)
    policy = policy_summary(all_detail)
    pair = same_sequence_detail(all_detail)
    pair_summary = same_sequence_summary(pair)
    policy_metrics = aggregate_policy_metrics(all_detail)
    normalized_metrics, normalization_scales = apply_common_monthly_normalization(policy_metrics)
    scores = score_grid(normalized_metrics)
    transitions = rank_transitions(scores)
    disagreement, regret = preferred_policy_and_regret(scores)
    bootstrap = cluster_bootstrap_same_sequence(pair, args.bootstrap_reps, args.seed)
    policy_bootstrap = cluster_bootstrap_policy_metrics(
        all_detail,
        args.bootstrap_reps,
        args.seed,
    )

    validation.to_csv(args.out_dir / "full_window_validation.csv", index=False)
    policy.to_csv(args.out_dir / "full_window_policy_summary.csv", index=False)
    pair_summary.to_csv(args.out_dir / "full_window_same_sequence_summary.csv", index=False)
    normalized_metrics.to_csv(args.out_dir / "full_window_common_normalized_policy_metrics.csv", index=False)
    normalization_scales.to_csv(args.out_dir / "full_window_common_normalization_scales.csv", index=False)
    scores.to_csv(args.out_dir / "full_window_policy_scores.csv", index=False)
    transitions.to_csv(args.out_dir / "full_window_policy_rank_transitions.csv", index=False)
    disagreement.to_csv(args.out_dir / "full_window_preferred_policy_disagreement.csv", index=False)
    regret.to_csv(args.out_dir / "full_window_policy_selection_regret.csv", index=False)
    bootstrap.to_csv(args.out_dir / "full_window_operating_date_cluster_bootstrap.csv", index=False)
    policy_bootstrap.to_csv(
        args.out_dir / "full_window_policy_metric_cluster_bootstrap.csv",
        index=False,
    )
    write_markdown(
        out_dir=args.out_dir,
        validation=validation,
        same_sequence=pair_summary,
        bootstrap=bootstrap,
        policy_bootstrap=policy_bootstrap,
        disagreement=disagreement,
        regret=regret,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
