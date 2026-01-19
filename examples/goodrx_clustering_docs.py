import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel, Field
from tqdm import tqdm

from synthefy import SynthefyAPIClient

np.random.seed(42)

NUM_GROUPS_FOR_ANALYSIS = 10
HISTORY_RATIO = 0.8
TIMESTAMP_COL = "week_start"
GROUP_COL = "drug_name"
TARGET_COL = "net_transactions"
METADATA_COLS = ["unit_ingredient_cost", "is_holiday", "holiday_weight"]

API_CLIENT = SynthefyAPIClient(
    api_key="e91c91c95faa4a9f029b7ed3e0a8a980f86469b7eb933419cc18d6246a46c3a1"
)


class CrossDrugForecastResult(BaseModel):
    """Data model for cross-drug forecasting results."""

    target_drug: str = Field(..., description="Target drug name")

    results: Dict[str, float] = Field(
        ..., description="Results of the forecast"
    )


def load_data_df_with_one_pharmacy():
    data_df = pd.read_parquet(
        "s3://goodrx-poc-datasets/GoodRxData/processed_all_pharm_bucket_weekly_agg_all.parquet"
    )
    data_df = data_df.sort_values("week_start").reset_index(drop=True)

    # To make the data smaller, lets only look at store 1
    data_df = data_df[
        data_df["parent_company_hash"]
        == "a2d6ddd9262d5d9b0e259e7d21a451b59540b37beae9c9cb767acebc7f4eec62"
    ]

    # Convert days_supply_bucket to int
    data_df["days_supply_bucket"] = data_df["days_supply_bucket"].astype(int)
    data_df = data_df[data_df["days_supply_bucket"] == 30]

    # Drop parent_company_hash and days_supply_bucket
    data_df = data_df.drop(
        columns=["parent_company_hash", "days_supply_bucket"]
    )
    data_df = data_df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    return data_df


def calculate_mape(y_true, y_pred, epsilon=1e-10):
    """Calculate Mean Absolute Percentage Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = np.abs(y_true) > epsilon
    if not np.any(mask):
        return np.inf

    mape = float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    )
    return mape


def build_cross_drug_forecast_results(data_df, history_ratio=HISTORY_RATIO):
    """Build list of cross-drug forecasting results with one result per drug containing a dictionary of results."""
    drug_ids = data_df[GROUP_COL].unique()
    results = []
    baseline_results = {}

    print("Building cross-drug forecast results...")
    total_drugs = len(drug_ids)
    with tqdm(total=total_drugs, desc="Cross-group forecasts") as pbar:
        for i, drug_id_i in enumerate(drug_ids):
            drug_results = {}

            for j, drug_id_j in enumerate(drug_ids):
                if i == j:
                    continue

                # Get data for both drugs
                drug_i_data = data_df[
                    data_df[GROUP_COL] == drug_id_i
                ].sort_values(TIMESTAMP_COL)
                drug_j_data = data_df[
                    data_df[GROUP_COL] == drug_id_j
                ].sort_values(TIMESTAMP_COL)

                # Split drug i's data (80% history, 20% target) based on timestamp
                split_idx_i = int(len(drug_i_data) * history_ratio)
                split_timestamp_i = drug_i_data.iloc[split_idx_i][TIMESTAMP_COL]

                hist_i = drug_i_data[
                    drug_i_data[TIMESTAMP_COL] < split_timestamp_i
                ]
                target_i = drug_i_data[
                    drug_i_data[TIMESTAMP_COL] >= split_timestamp_i
                ]

                # Use drug i's timestamp to split drug j's data for alignment
                hist_j = drug_j_data[
                    drug_j_data[TIMESTAMP_COL] < split_timestamp_i
                ]
                target_j = drug_j_data[
                    drug_j_data[TIMESTAMP_COL] >= split_timestamp_i
                ]

                # Merge group j's target with group i's data
                j_target = target_j[[TIMESTAMP_COL, TARGET_COL]].rename(
                    columns={TARGET_COL: f"{TARGET_COL}_leaked"}
                )
                merged_target = target_i.merge(
                    j_target, on=TIMESTAMP_COL, how="inner"
                )

                # Add leaked column to history for forecasting
                j_hist = hist_j[[TIMESTAMP_COL, TARGET_COL]].rename(
                    columns={TARGET_COL: f"{TARGET_COL}_leaked"}
                )
                hist_i_with_leak = hist_i.merge(
                    j_hist, on=TIMESTAMP_COL, how="inner"
                )

                # Make forecast using API with leaks and metadata
                try:
                    forecast_dfs = API_CLIENT.forecast_dfs(
                        history_dfs=[hist_i_with_leak],
                        target_dfs=[merged_target],
                        target_col=TARGET_COL,
                        timestamp_col=TIMESTAMP_COL,
                        metadata_cols=METADATA_COLS + [f"{TARGET_COL}_leaked"],
                        leak_cols=[f"{TARGET_COL}_leaked"],
                        model="sfm-moe-v1",
                    )

                    predictions = forecast_dfs[0][TARGET_COL].values
                    ground_truth = merged_target[TARGET_COL].values

                    # Calculate MAPE
                    mape = calculate_mape(ground_truth, predictions)
                    drug_results[drug_id_j] = float(mape)

                except Exception as e:
                    print(
                        f"Forecast failed for drugs {drug_id_i}, {drug_id_j}: {e}"
                    )
                    drug_results[drug_id_j] = float(np.inf)

                # Also calculate baseline forecast (no leaks, no metadata) for comparison
                if j == 0:  # Only do this once per target drug
                    try:
                        baseline_forecast = API_CLIENT.forecast_dfs(
                            history_dfs=[hist_i],
                            target_dfs=[target_i],
                            target_col=TARGET_COL,
                            timestamp_col=TIMESTAMP_COL,
                            metadata_cols=[],
                            leak_cols=[],
                            model="sfm-moe-v1",
                        )

                        baseline_predictions = baseline_forecast[0][
                            TARGET_COL
                        ].values
                        baseline_ground_truth = target_i[TARGET_COL].values
                        baseline_mape = calculate_mape(
                            baseline_ground_truth, baseline_predictions
                        )
                        baseline_results[drug_id_i] = float(baseline_mape)

                    except Exception as e:
                        print(
                            f"Baseline forecast failed for drug {drug_id_i}: {e}"
                        )
                        baseline_results[drug_id_i] = float(np.inf)

            # Create result instance for this drug
            result = CrossDrugForecastResult(
                target_drug=drug_id_i, results=drug_results
            )
            results.append(result)
            pbar.update(1)

    return results, baseline_results


def create_individual_bar_plots(forecast_results, data_df):
    """Create 6 vertical charts for each drug: 1 bar plot + 4 time series plots + 1 combined plot."""

    for result in forecast_results:
        target_drug = result.target_drug

        # Create figure with 6 vertical subplots
        fig, axes = plt.subplots(6, 1, figsize=(16, 20))

        # Get leaked drugs and MAPE values
        leaked_drugs = list(result.results.keys())
        mape_values = list(result.results.values())

        # Sort by MAPE (best to worst)
        sorted_pairs = sorted(
            zip(leaked_drugs, mape_values), key=lambda x: x[1]
        )
        sorted_drugs, sorted_mape = zip(*sorted_pairs)

        # 1. Bar plot (top chart)
        ax1 = axes[0]
        bars = ax1.bar(
            range(len(sorted_drugs)), sorted_mape, color="steelblue", alpha=0.7
        )
        ax1.set_xlabel("Leaked Drug (ranked by MAPE)")
        ax1.set_ylabel("MAPE (%)")
        ax1.set_title(f"MAPE Performance - Drug {target_drug} as Target")
        ax1.set_xticks(range(len(sorted_drugs)))
        ax1.set_xticklabels(sorted_drugs, rotation=45)
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (bar, mape) in enumerate(zip(bars, sorted_mape)):
            if not np.isnan(mape) and mape != np.inf:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f"{mape:.1f}",
                    ha="center",
                    va="bottom",
                )

        # 2. Target drug time series
        ax2 = axes[1]
        target_data = data_df[data_df[GROUP_COL] == target_drug].sort_values(
            TIMESTAMP_COL
        )
        ax2.plot(
            target_data[TIMESTAMP_COL],
            target_data[TARGET_COL],
            linewidth=2,
            color="blue",
            label=f"Target Drug {target_drug}",
        )
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Net Transactions")
        ax2.set_title(f"Target Drug {target_drug} Time Series")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.get_xticklabels(), rotation=45)

        # 3-5. Top 3 best performing leaked drugs time series
        valid_pairs = [
            (drug, mape)
            for drug, mape in sorted_pairs
            if not np.isnan(mape) and mape != np.inf
        ]
        top_3_drugs = valid_pairs[:3]  # Get top 3 best performing

        colors_ts = ["green", "orange", "red"]
        for i, (drug_id, mape) in enumerate(top_3_drugs):
            ax = axes[2 + i]
            drug_data = data_df[data_df[GROUP_COL] == drug_id].sort_values(
                TIMESTAMP_COL
            )
            ax.plot(
                drug_data[TIMESTAMP_COL],
                drug_data[TARGET_COL],
                linewidth=2,
                color=colors_ts[i],
                label=f"Drug {drug_id} (MAPE: {mape:.2f}%)",
            )
            ax.set_xlabel("Date")
            ax.set_ylabel("Net Transactions")
            ax.set_title(
                f"Best Leaked Drug #{i + 1}: {drug_id} (MAPE: {mape:.2f}%)"
            )
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)
            plt.setp(ax.get_xticklabels(), rotation=45)

        # 6. Combined time series plot
        ax6 = axes[5]

        # Plot target drug (normalized)
        target_data = data_df[data_df[GROUP_COL] == target_drug].sort_values(
            TIMESTAMP_COL
        )
        target_values = target_data[TARGET_COL].values
        target_normalized = (target_values - target_values.min()) / (
            target_values.max() - target_values.min()
        )
        ax6.plot(
            target_data[TIMESTAMP_COL],
            target_normalized,
            linewidth=3,
            color="blue",
            label=f"Target Drug {target_drug}",
        )

        # Plot top 3 leaked drugs (normalized)
        colors_combined = ["green", "orange", "red"]
        for i, (drug_id, mape) in enumerate(top_3_drugs):
            drug_data = data_df[data_df[GROUP_COL] == drug_id].sort_values(
                TIMESTAMP_COL
            )
            drug_values = drug_data[TARGET_COL].values
            drug_normalized = (drug_values - drug_values.min()) / (
                drug_values.max() - drug_values.min()
            )
            ax6.plot(
                drug_data[TIMESTAMP_COL],
                drug_normalized,
                linewidth=2,
                color=colors_combined[i],
                alpha=0.8,
                label=f"Drug {drug_id} (MAPE: {mape:.2f}%)",
            )

        ax6.set_xlabel("Date")
        ax6.set_ylabel("Normalized Net Transactions (0-1)")
        ax6.set_title(
            f"Combined Time Series - Target Drug {target_drug} vs Top 3 Leaked Drugs (Normalized)"
        )
        ax6.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax6.grid(True, alpha=0.3)
        plt.setp(ax6.get_xticklabels(), rotation=45)

        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(hspace=0.4)

        # Create plots directory if it doesn't exist
        os.makedirs("plots", exist_ok=True)

        # Save the plot
        filename = f"plots/drug_analysis_{target_drug.replace(' ', '_').replace('/', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {filename}")

        plt.close()

        # Print summary for this drug
        valid_mape = [
            mape
            for mape in sorted_mape
            if not np.isnan(mape) and mape != np.inf
        ]
        if valid_mape:
            print(f"\nDrug {target_drug} Summary:")
            print(
                f"  Best leaked drug: {sorted_drugs[0]} (MAPE: {sorted_mape[0]:.2f}%)"
            )
            print(
                f"  Worst leaked drug: {sorted_drugs[-1]} (MAPE: {sorted_mape[-1]:.2f}%)"
            )
            print(f"  Average MAPE: {np.mean(valid_mape):.2f}%")


def main():
    # Load the data
    data_df = load_data_df_with_one_pharmacy()

    # pick 10 random drug_ids for the analysis
    group_ids = np.random.choice(
        data_df[GROUP_COL].unique(), NUM_GROUPS_FOR_ANALYSIS, replace=False
    )
    data_df = data_df[data_df[GROUP_COL].isin(group_ids)]

    # Build cross-drug forecast results using helper function (now returns both)
    forecast_results, baseline_results = build_cross_drug_forecast_results(
        data_df, history_ratio=HISTORY_RATIO
    )

    # Compare univariate vs multivariate performance
    print("\nComparing univariate vs multivariate performance...")
    baseline_mape_values = [
        mape
        for mape in baseline_results.values()
        if mape != np.inf and not np.isnan(mape)
    ]
    multivariate_mape_values = []
    for result in forecast_results:
        # Get the best MAPE for this target drug from all its cross-drug results
        valid_mape_values = [
            v
            for v in result.results.values()
            if v != np.inf and not np.isnan(v)
        ]
        if valid_mape_values:
            best_mape = min(valid_mape_values)
            multivariate_mape_values.append(best_mape)

    if baseline_mape_values and multivariate_mape_values:
        baseline_avg = np.mean(baseline_mape_values)
        multivariate_avg = np.mean(multivariate_mape_values)
        improvement = baseline_avg - multivariate_avg
        improvement_pct = (improvement / baseline_avg) * 100

        print("\nCOMPARISON:")
        print(f"Univariate (no leaks, no metadata): {baseline_avg:.2f}% MAPE")
        print(
            f"Multivariate (with leaks + metadata): {multivariate_avg:.2f}% MAPE"
        )
        print(
            f"Improvement: {improvement:.2f} percentage points ({improvement_pct:.1f}%)"
        )

    # Create individual bar plots for each drug
    create_individual_bar_plots(forecast_results, data_df)

    return data_df, forecast_results, baseline_results


if __name__ == "__main__":
    main()
