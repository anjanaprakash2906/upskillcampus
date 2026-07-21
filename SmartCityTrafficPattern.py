import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


WORKSPACE_ROOT = Path(__file__).resolve().parent
DATA_DIR = WORKSPACE_ROOT / "smart-city-traffic-patterns"
TRAIN_PATH = DATA_DIR / "train_aWnotuB.csv"
TEST_PATH = DATA_DIR / "datasets_8494_11879_test_BdBKkAj.csv"
OUTPUT_PATH = WORKSPACE_ROOT / "smartcity_predictions.csv"
REPORT_PATH = WORKSPACE_ROOT / "smartcity_report.md"
PLOT_PATH = WORKSPACE_ROOT / "smartcity_forecast_plot.png"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("DateTime").reset_index(drop=True)

    df["hour"] = df["DateTime"].dt.hour
    df["day_of_week"] = df["DateTime"].dt.dayofweek
    df["day_of_month"] = df["DateTime"].dt.day
    df["month"] = df["DateTime"].dt.month
    df["week_of_year"] = df["DateTime"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["quarter"] = ((df["month"] - 1) // 3) + 1

    holiday_dates = {
        "01-01",
        "01-26",
        "07-04",
        "11-11",
        "12-25",
    }
    df["is_holiday"] = df["DateTime"].dt.strftime("%m-%d").isin(holiday_dates).astype(int)
    df["is_month_start"] = df["DateTime"].dt.day.eq(1).astype(int)
    df["is_month_end"] = df["DateTime"].dt.is_month_end.astype(int)
    df["is_quarter_start"] = (
        df["DateTime"].dt.day.eq(1) & df["DateTime"].dt.month.isin([1, 4, 7, 10])
    ).astype(int)

    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_day"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_day"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["sin_week"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["cos_week"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lag_1"] = df["Vehicles"].shift(1)
    df["lag_24"] = df["Vehicles"].shift(24)
    df["lag_168"] = df["Vehicles"].shift(168)
    return df


def prepare_training_frame(train_junction: pd.DataFrame) -> pd.DataFrame:
    frame = add_time_features(train_junction)
    frame = add_lag_features(frame)

    fill_value = frame["Vehicles"].median()
    frame["lag_1"] = frame["lag_1"].fillna(fill_value)
    frame["lag_24"] = frame["lag_24"].fillna(fill_value)
    frame["lag_168"] = frame["lag_168"].fillna(fill_value)
    return frame


def build_feature_columns() -> list[str]:
    return [
        "hour",
        "day_of_week",
        "day_of_month",
        "month",
        "week_of_year",
        "is_weekend",
        "is_holiday",
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "quarter",
        "sin_hour",
        "cos_hour",
        "sin_day",
        "cos_day",
        "sin_month",
        "cos_month",
        "sin_week",
        "cos_week",
        "lag_1",
        "lag_24",
        "lag_168",
    ]


def train_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    train_features = add_time_features(train_df)
    global_mean = float(train_features["Vehicles"].mean())

    lookup = {}
    base_profile = train_features.groupby(
        ["Junction", "hour", "day_of_week", "is_weekend", "is_holiday", "month", "quarter"]
    )["Vehicles"].mean().reset_index(name="predicted")
    for _, row in base_profile.iterrows():
        lookup[(int(row["Junction"]), int(row["hour"]), int(row["day_of_week"]), int(row["is_weekend"]), int(row["is_holiday"]), int(row["month"]), int(row["quarter"]))] = float(row["predicted"])

    fallback_profiles = [
        ["Junction", "hour", "day_of_week", "is_weekend", "is_holiday"],
        ["Junction", "hour", "day_of_week", "is_weekend"],
        ["Junction", "hour", "day_of_week"],
        ["Junction", "hour"],
        ["Junction"],
    ]

    fallback_lookups = []
    for columns in fallback_profiles:
        profile = train_features.groupby(columns)["Vehicles"].mean().reset_index(name="predicted")
        lookup_map = {}
        for _, row in profile.iterrows():
            key = tuple(int(row[column]) if pd.api.types.is_integer_dtype(type(row[column])) else row[column] for column in columns)
            lookup_map[key] = float(row["predicted"])
        fallback_lookups.append(lookup_map)

    predictions = []
    metrics = {}

    for junction in sorted(train_df["Junction"].unique()):
        junction_train = train_features.loc[train_features["Junction"] == junction].copy()
        val_split = max(1, int(len(junction_train) * 0.9))
        val_actual = junction_train.iloc[val_split:]["Vehicles"].to_numpy()
        val_pred = []
        for _, row in junction_train.iloc[val_split:].iterrows():
            val_pred.append(predict_from_lookup(row, lookup, fallback_lookups, global_mean))
        val_pred = np.array(val_pred, dtype=float)
        metrics[f"junction_{junction}"] = {
            "mae": round(float(mean_absolute_error(val_actual, val_pred)), 3),
            "rmse": round(float(math.sqrt(mean_squared_error(val_actual, val_pred))), 3),
        }

    for _, row in test_df.sort_values("DateTime").iterrows():
        predictions.append((row["ID"], predict_from_lookup(row, lookup, fallback_lookups, global_mean)))

    prediction_df = pd.DataFrame(predictions, columns=["ID", "Vehicles"])
    prediction_df = prediction_df.sort_values("ID").reset_index(drop=True)
    return prediction_df, metrics


def predict_from_lookup(
    row: pd.Series,
    lookup: dict[tuple, float],
    fallback_lookups: list[dict[tuple, float]],
    global_mean: float,
) -> float:
    date_time = row["DateTime"] if "DateTime" in row.index else row["datetime"]
    if isinstance(date_time, str):
        date_time = pd.to_datetime(date_time)

    junction = int(row["Junction"])
    hour = int(date_time.hour)
    day_of_week = int(date_time.dayofweek)
    is_weekend = int(date_time.dayofweek >= 5)
    is_holiday = int(date_time.strftime("%m-%d") in {"01-01", "01-26", "07-04", "11-11", "12-25"})
    month = int(date_time.month)
    quarter = ((month - 1) // 3) + 1

    keys = [
        (junction, hour, day_of_week, is_weekend, is_holiday, month, quarter),
        (junction, hour, day_of_week, is_weekend, is_holiday),
        (junction, hour, day_of_week, is_weekend),
        (junction, hour, day_of_week),
        (junction, hour),
        (junction,),
    ]

    for key in keys:
        if key in lookup:
            return float(lookup[key])

    for lookup_map in fallback_lookups:
        for key in keys:
            if len(key) <= len(lookup_map) and tuple(key[: len(lookup_map)]) in lookup_map:
                return float(lookup_map[tuple(key[: len(lookup_map)])])

    return global_mean


def make_forecast_plot(train_df: pd.DataFrame, test_df: pd.DataFrame, prediction_df: pd.DataFrame) -> None:
    merged = test_df.merge(prediction_df, on="ID", how="left")
    merged = merged.sort_values(["Junction", "DateTime"]).reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    axes = axes.flatten()

    for idx, junction in enumerate(sorted(train_df["Junction"].unique())):
        train_series = (
            train_df.loc[train_df["Junction"] == junction, ["DateTime", "Vehicles"]]
            .set_index("DateTime")
            .resample("D")
            .mean()
        )
        forecast_series = (
            merged.loc[merged["Junction"] == junction, ["DateTime", "Vehicles"]]
            .set_index("DateTime")
            .resample("D")
            .mean()
        )

        ax = axes[idx]
        ax.plot(train_series.index, train_series["Vehicles"], color="tab:orange", linewidth=1.5, label="Historical traffic")
        ax.plot(forecast_series.index, forecast_series["Vehicles"], color="tab:blue", linewidth=1.5, label="Forecast")
        ax.set_title(f"Junction {junction}")
        ax.set_ylabel("Vehicles")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left")

    for axis in axes[len(sorted(train_df["Junction"].unique())) :]:
        axis.axis("off")

    fig.suptitle("Smart City Traffic Forecast by Junction", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(PLOT_PATH, dpi=220)
    plt.close(fig)


def write_report(test_df: pd.DataFrame, prediction_df: pd.DataFrame, metrics: dict) -> None:
    merged = test_df.merge(prediction_df, on="ID", how="left")

    report_lines = []
    report_lines.append("# Smart City Traffic Forecast Report")
    report_lines.append("")
    report_lines.append("## Overview")
    report_lines.append(f"- Total predictions: {len(merged)}")
    report_lines.append(f"- Average predicted vehicles: {merged['Vehicles'].mean():.2f}")
    report_lines.append(f"- Peak predicted vehicles: {merged['Vehicles'].max():.2f}")
    report_lines.append(f"- Lowest predicted vehicles: {merged['Vehicles'].min():.2f}")
    report_lines.append("")
    report_lines.append("## Forecast Plot")
    report_lines.append("")
    report_lines.append(f"![Traffic forecast plot]({PLOT_PATH.name})")
    report_lines.append("")
    report_lines.append("## Validation Metrics")
    report_lines.append("| Junction | MAE | RMSE |")
    report_lines.append("|---|---:|---:|")
    for junction_key, values in metrics.items():
        junction_name = junction_key.replace("junction_", "Junction ")
        report_lines.append(f"| {junction_name} | {values['mae']:.3f} | {values['rmse']:.3f} |")
    report_lines.append("")
    report_lines.append("## Highest Traffic Predictions")
    report_lines.append("| Rank | ID | Junction | DateTime | Predicted Vehicles |")
    report_lines.append("|---|---|---|---|---:|")
    top_predictions = merged.sort_values("Vehicles", ascending=False).head(10)
    for rank, row in enumerate(top_predictions.itertuples(index=False), start=1):
        report_lines.append(
            f"| {rank} | {row.ID} | {row.Junction} | {row.DateTime} | {row.Vehicles:.2f} |"
        )

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH, parse_dates=["DateTime"])
    test_df = pd.read_csv(TEST_PATH, parse_dates=["DateTime"])

    prediction_df, metrics = train_and_predict(train_df, test_df)
    prediction_df.to_csv(OUTPUT_PATH, index=False)
    make_forecast_plot(train_df, test_df, prediction_df)
    write_report(test_df, prediction_df, metrics)

    print("\n========================================")
    print("SMART CITY TRAFFIC FORECAST REPORT")
    print("========================================")
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Prediction file: {OUTPUT_PATH}")
    print(f"Summary report: {REPORT_PATH}")
    print(f"Forecast plot: {PLOT_PATH}")
    print("\nValidation metrics:")
    for junction, values in metrics.items():
        junction_name = junction.replace("junction_", "Junction ")
        print(f"  - {junction_name}: MAE={values['mae']}, RMSE={values['rmse']}")
    print("\nDone. Open the report and prediction file to review the results.")


if __name__ == "__main__":
    main()
