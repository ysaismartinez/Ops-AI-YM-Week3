"""
Data Quality Validation Framework Template
"""

import pandas as pd
import numpy as np
from typing import Dict


class DataQualityValidator:
    """Validates data against quality expectations."""

    def __init__(self, baseline_df: pd.DataFrame = None):
        self.baseline = baseline_df
        self.issues = []

    def validate(self, df: pd.DataFrame) -> Dict:
        """Run all validation checks."""
        self.issues = []

        self.check_schema(df)
        self.check_null_rates(df)
        self.check_value_ranges(df)
        self.check_duplicates(df)
        self.check_distributions(df)
        self.check_volume(df)

        return {
            "is_valid": len(self.issues) == 0,
            "num_issues": len(self.issues),
            "issues": self.issues,
        }

    def check_null_rates(self, df: pd.DataFrame):
        """Check if any column has excessive nulls."""
        if self.baseline is None:
            return

        baseline_nulls = self.baseline.isnull().mean()
        current_nulls = df.isnull().mean()

        for col in df.columns:
            if col in baseline_nulls:
                increase = current_nulls[col] - baseline_nulls[col]
                if increase > 0.01:
                    self._add_issue(
                        "missing_values",
                        "high",
                        f"Null rate increased significantly in column {col}",
                        column=col,
                        baseline_null_rate=float(baseline_nulls[col]),
                        current_null_rate=float(current_nulls[col]),
                        increase=float(increase),
                    )

    def check_value_ranges(self, df: pd.DataFrame):
        """Check if values fall within expected ranges."""

        if "trip_count" in df.columns:
            negative_count = int((df["trip_count"] < 0).sum())

            if negative_count > 0:
                self._add_issue(
                    "invalid_value_range",
                    "critical",
                    "trip_count contains negative values, which are logically invalid.",
                    count=negative_count,
                    column="trip_count",
                )

            if self.baseline is not None and "trip_count" in self.baseline.columns:
                baseline_max = self.baseline["trip_count"].max()
                extreme_count = int((df["trip_count"] > baseline_max * 10).sum())

                if extreme_count > 0:
                    self._add_issue(
                        "extreme_outliers",
                        "critical",
                        "trip_count contains extreme outliers compared with baseline.",
                        count=extreme_count,
                        column="trip_count",
                        baseline_max=float(baseline_max),
                        current_max=float(df["trip_count"].max()),
                    )

        if "hour" in df.columns:
            bad_hours = int(((df["hour"] < 0) | (df["hour"] > 23)).sum())
            if bad_hours > 0:
                self._add_issue(
                    "invalid_value_range",
                    "high",
                    "hour contains values outside expected range 0-23.",
                    count=bad_hours,
                    column="hour",
                )

        if "dayofweek" in df.columns:
            bad_days = int(((df["dayofweek"] < 0) | (df["dayofweek"] > 6)).sum())
            if bad_days > 0:
                self._add_issue(
                    "invalid_value_range",
                    "high",
                    "dayofweek contains values outside expected range 0-6.",
                    count=bad_days,
                    column="dayofweek",
                )

    def check_distributions(self, df: pd.DataFrame):
        """Check if data distribution matches baseline."""
        if self.baseline is None:
            return

        if "trip_count" not in df.columns or "trip_count" not in self.baseline.columns:
            return

        baseline_mean = self.baseline["trip_count"].mean()
        baseline_std = self.baseline["trip_count"].std()
        current_mean = df["trip_count"].mean()
        current_std = df["trip_count"].std()

        mean_shift_ratio = current_mean / baseline_mean if baseline_mean != 0 else np.inf
        std_shift_ratio = current_std / baseline_std if baseline_std != 0 else np.inf

        if mean_shift_ratio > 2 or std_shift_ratio > 5:
            self._add_issue(
                "distribution_shift",
                "high",
                "trip_count distribution shifted significantly from baseline.",
                column="trip_count",
                baseline_mean=float(baseline_mean),
                current_mean=float(current_mean),
                mean_shift_ratio=float(mean_shift_ratio),
                baseline_std=float(baseline_std),
                current_std=float(current_std),
                std_shift_ratio=float(std_shift_ratio),
            )

    def check_duplicates(self, df: pd.DataFrame):
        """Check for duplicate rows."""
        duplicate_count = int(df.duplicated().sum())

        if duplicate_count > 0:
            self._add_issue(
                "duplicate_records",
                "high",
                "Dataset contains duplicate rows.",
                count=duplicate_count,
            )

    def check_schema(self, df: pd.DataFrame):
        """Check that required columns exist with correct types."""
        if self.baseline is None:
            return

        baseline_cols = list(self.baseline.columns)
        current_cols = list(df.columns)

        if baseline_cols != current_cols:
            self._add_issue(
                "schema_mismatch",
                "critical",
                "Current dataset columns do not match baseline columns.",
                baseline_columns=baseline_cols,
                current_columns=current_cols,
            )

        dtype_mismatches = []

        for col in self.baseline.columns:
            if col in df.columns and df[col].dtype != self.baseline[col].dtype:
                dtype_mismatches.append({
                    "column": col,
                    "baseline_dtype": str(self.baseline[col].dtype),
                    "current_dtype": str(df[col].dtype),
                })

        if dtype_mismatches:
            self._add_issue(
                "datatype_mismatch",
                "critical",
                "One or more column datatypes do not match baseline.",
                mismatches=dtype_mismatches,
            )

    def check_volume(self, df: pd.DataFrame):
        """Check for major dataset volume reduction."""
        if self.baseline is None:
            return

        baseline_rows = len(self.baseline)
        current_rows = len(df)

        if baseline_rows == 0:
            return

        ratio = current_rows / baseline_rows

        if ratio < 0.10:
            self._add_issue(
                "dataset_volume_reduction",
                "medium",
                "Current dataset has dramatically fewer rows than baseline.",
                baseline_rows=baseline_rows,
                current_rows=current_rows,
                ratio=float(ratio),
            )

    def _add_issue(
        self,
        issue_type: str,
        severity: str,
        description: str,
        count: int = None,
        **details
    ):
        issue = {
            "type": issue_type,
            "severity": severity,
            "description": description,
            "count": count,
            **details,
        }
        self.issues.append(issue)


def compare_distributions(
    baseline: pd.Series, current: pd.Series, threshold: float = 2.0
) -> bool:
    """Compare distributions using mean shift."""
    baseline_mean = baseline.mean()
    current_mean = current.mean()

    if baseline_mean == 0:
        return False

    ratio = current_mean / baseline_mean
    return ratio > threshold or ratio < (1 / threshold)


def detect_outliers(
    series: pd.Series, baseline_series: pd.Series = None, sigma: float = 3.0
) -> pd.Series:
    """Detect outliers in a numeric series."""
    reference = baseline_series if baseline_series is not None else series

    mean = reference.mean()
    std = reference.std()

    if std == 0:
        return pd.Series(False, index=series.index)

    return abs(series - mean) > sigma * std


if __name__ == "__main__":
    full_df = pd.read_parquet("data/demand_enriched_corrupted.parquet")
    full_df["time_bucket"] = pd.to_datetime(full_df["time_bucket"])

    baseline_df = full_df[full_df["time_bucket"] < "2026-01-16"]
    corrupted_df = full_df[full_df["time_bucket"] >= "2026-01-16"]

    validator = DataQualityValidator(baseline_df=baseline_df)
    result = validator.validate(corrupted_df)

    print("Is valid:", result["is_valid"])
    print("Number of issues:", result["num_issues"])

    for issue in result["issues"]:
        print()
        print(issue)

    if not result["is_valid"]:
        raise SystemExit(1)