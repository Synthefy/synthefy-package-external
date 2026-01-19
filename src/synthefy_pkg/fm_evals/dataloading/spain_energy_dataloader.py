from typing import Iterator

import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class SpainEnergyDataloader(BaseEvalDataloader):
    # Discrete metadata columns
    """
    DISCRETE_METADATA = [
        "Barcelona_weather_description",
        "Bilbao_weather_description",
        "Madrid_weather_description",
        "Seville_weather_description",
        "Valencia_weather_description",
        " Barcelona_weather_icon",
        "Bilbao_weather_icon",
        "Madrid_weather_icon",
        "Seville_weather_icon",
        "Valencia_weather_icon",
        " Barcelona_weather_id",
        "Bilbao_weather_id",
        "Madrid_weather_id",
        "Seville_weather_id",
        "Valencia_weather_id",
        " Barcelona_weather_main",
        "Bilbao_weather_main",
        "Madrid_weather_main",
        "Seville_weather_main",
        "Valencia_weather_main",
    ]
    """

    DISCRETE_METADATA = []

    # Continuous metadata columns
    CONTINUOUS_METADATA = [
        # " Barcelona_clouds_all",
        # "Bilbao_clouds_all",
        # "Madrid_clouds_all",
        # "Seville_clouds_all",
        # "Valencia_clouds_all",
        # " Barcelona_humidity",
        # "Bilbao_humidity",
        # "Madrid_humidity",
        # "Seville_humidity",
        # "Valencia_humidity",
        # " Barcelona_pressure",
        # "Bilbao_pressure",
        # "Madrid_pressure",
        # "Seville_pressure",
        # "Valencia_pressure",
        # " Barcelona_rain_1h",
        # "Bilbao_rain_1h",
        # "Madrid_rain_1h",
        # "Seville_rain_1h",
        # "Valencia_rain_1h",
        # " Barcelona_rain_3h",
        # "Bilbao_rain_3h",
        # "Madrid_rain_3h",
        # "Seville_rain_3h",
        # "Valencia_rain_3h",
        # "Bilbao_snow_3h",
        # "Madrid_snow_3h",
        # "Valencia_snow_3h",
        " Barcelona_temp",
        "Bilbao_temp",
        "Madrid_temp",
        "Seville_temp",
        "Valencia_temp",
        # " Barcelona_temp_max",
        # "Bilbao_temp_max",
        # "Madrid_temp_max",
        # "Seville_temp_max",
        # "Valencia_temp_max",
        # " Barcelona_temp_min",
        # "Bilbao_temp_min",
        # "Madrid_temp_min",
        # "Seville_temp_min",
        # "Valencia_temp_min",
        # " Barcelona_wind_deg",
        # "Bilbao_wind_deg",
        # "Madrid_wind_deg",
        # "Seville_wind_deg",
        # "Valencia_wind_deg",
        # " Barcelona_wind_speed",
        # "Bilbao_wind_speed",
        # "Madrid_wind_speed",
        # "Seville_wind_speed",
        # "Valencia_wind_speed",
        # "generation biomass",
        # "generation fossil brown coal/lignite",
        # "generation fossil gas",
        # "generation fossil hard coal",
        # "generation fossil oil",
        # "generation hydro pumped storage consumption",
        # "generation hydro run-of-river and poundage",
        # "generation hydro water reservoir",
        # "generation nuclear",
        # "generation other",
        # "generation other renewable",
        # "generation solar",
        # "generation waste",
        # "generation wind onshore",
        # "forecast solar day ahead",
        # "forecast wind onshore day ahead",
    ]

    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.df = self._load_data()
        self.df = self._encode_discrete_metadata()
        self.backtesting_batch = self._get_backtesting_batch()

    def _load_data(self) -> pd.DataFrame:
        data_path = "s3://synthefy-core/energy_demand/energy_demand.parquet"
        return pd.read_parquet(data_path)

    def _encode_discrete_metadata(self) -> pd.DataFrame:
        for col in self.DISCRETE_METADATA:
            unique_values = self.df[col].unique()
            value_to_position = {
                val: pos for pos, val in enumerate(unique_values)
            }
            self.df[col] = self.df[col].map(value_to_position)
        return self.df

    def _get_backtesting_batch(self) -> EvalBatchFormat:
        to_return = EvalBatchFormat.from_dfs(
            dfs=[self.df],
            timestamp_col="time",
            num_target_rows=24 * 30,
            target_cols=["total load actual"],
            metadata_cols=self.DISCRETE_METADATA + self.CONTINUOUS_METADATA,
            forecast_window=24,
            stride=24,
        )

        if to_return is None:
            raise ValueError("No valid backtesting batch found")

        return to_return

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Iterator[EvalBatchFormat]:
        yield self.backtesting_batch


if __name__ == "__main__":
    dataloader = SpainEnergyDataloader()
    for batch in dataloader:
        print(batch)
        break
