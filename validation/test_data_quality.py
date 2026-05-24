"""
Data Quality Validation Tests
"""

import pytest
import pandas as pd
from validation.check_data_quality import DataQualityValidator


DATA_PATH = "data/demand_enriched_corrupted.parquet"
CUTOFF_DATE = "2026-01-16"


@pytest.fixture
def full_data():
    df = pd.read_parquet(DATA_PATH)
    df["time_bucket"] = pd.to_datetime(df["time_bucket"])
    return df


@pytest.fixture
def baseline_data(full_data):
    return full_data[full_data["time_bucket"] < CUTOFF_DATE].copy()


@pytest.fixture
def corrupted_data(full_data):
    return full_data[full_data["time_bucket"] >= CUTOFF_DATE].copy()


@pytest.fixture
def validator(baseline_data):
    return DataQualityValidator(baseline_df=baseline_data)


class TestBaselineData:
    def test_baseline_passes_validation(self, baseline_data):
        validator = DataQualityValidator(baseline_df=baseline_data)
        result = validator.validate(baseline_data)

        assert result["is_valid"], f"Baseline failed validation: {result['issues']}"


class TestDataQualityIssues:
    def test_corrupted_data_fails_validation(self, corrupted_data, validator):
        result = validator.validate(corrupted_data)

        assert not result["is_valid"]
        assert result["num_issues"] >= 2

    def test_detect_duplicate_records(self, corrupted_data, validator):
        result = validator.validate(corrupted_data)

        assert any(issue["type"] == "duplicate_records" for issue in result["issues"])

    def test_detect_extreme_outliers(self, corrupted_data, validator):
        result = validator.validate(corrupted_data)

        assert any(issue["type"] == "extreme_outliers" for issue in result["issues"])

    def test_detect_invalid_negative_trip_counts(self, corrupted_data, validator):
        result = validator.validate(corrupted_data)

        assert any(
            issue["type"] == "invalid_value_range"
            and issue.get("column") == "trip_count"
            for issue in result["issues"]
        )

    def test_detect_distribution_shift(self, corrupted_data, validator):
        result = validator.validate(corrupted_data)

        assert any(issue["type"] == "distribution_shift" for issue in result["issues"])

    def test_detect_dataset_volume_reduction(self, corrupted_data, validator):
        result = validator.validate(corrupted_data)

        assert any(
            issue["type"] == "dataset_volume_reduction"
            for issue in result["issues"]
        )


class TestGracefulDegradation:
    def test_validation_does_not_crash_with_bad_data(self, corrupted_data, validator):
        try:
            result = validator.validate(corrupted_data)
        except Exception as e:
            pytest.fail(f"Validation crashed on corrupted data: {e}")

        assert result is not None
        assert "is_valid" in result
        assert "issues" in result

    def test_bad_data_is_reported_not_silently_ignored(self, corrupted_data, validator):
        result = validator.validate(corrupted_data)

        assert not result["is_valid"]
        assert result["num_issues"] > 0