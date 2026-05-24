import pandas as pd

CUTOFF = pd.Timestamp("2026-01-16")

df = pd.read_parquet('data/demand_enriched_corrupted.parquet')
baseline = df[df['time_bucket'] < CUTOFF]   # clean historical window
corrupted = df[df['time_bucket'] >= CUTOFF]  # new potentially corrupted window

print(f"Baseline: {len(baseline)} rows")
print(f"Corrupted: {len(corrupted)} rows")
print(f"\nBaseline columns: {baseline.columns.tolist()}")
print(f"\nBaseline null rates:\n{baseline.isna().mean()}")
print(f"\nCorrupted null rates:\n{corrupted.isna().mean()}")
print("Baseline duplicates:", baseline.duplicated().sum())
print("Corrupted duplicates:", corrupted.duplicated().sum())
print(baseline["trip_count"].describe())
print(corrupted["trip_count"].describe())
print(corrupted[corrupted["trip_count"] > baseline["trip_count"].mean() * 10])
print(baseline.dtypes)
print(corrupted.dtypes)
print(baseline["time_bucket"].min(), baseline["time_bucket"].max())
print(corrupted["time_bucket"].min(), corrupted["time_bucket"].max())