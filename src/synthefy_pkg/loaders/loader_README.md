# Loaders enrich data loaded from standardized data into dataset parquets and metadata
The data Parquet files are stored in /home/data/enriched_datasets/

The loaders can load part of the dataset (for debugging) or all of the dataset
Currently two loaders are supported:
    - simple_loader.py
        - Description: Loads standardized datasets and breaks them up to one column per dataset without applying enrichment. Useful for univariate forecasting, and superclass for subsequent loaders. THe selection of which standardized datasets to load uses the complex filter logic. If that logic is not provided, takes the AND of all filters.
        - Run Example: python src/synthefy_pkg/loaders/simple_loader.py --config src/synthefy_pkg/loaders/loader_configs/CPI_fred.yaml --outname fred_univariate
        - See src/synthefy_pkg/loaders/loader_configs/loader_schema.yaml for a detailed schema of the config files

    - merge_loader.py
        - Description: Loads standardized datasets twice, first to get the main series and break them up, then again to enrich the dataset with metadata. Synchronizes the time index of the metadata to that of the main series, assuming canonical frequencies: Hourly, Daily, Weekly, Monthly, Quarterly, Yearly.
        - Run Example: python src/synthefy_pkg/loaders/merge_loader.py --config src/synthefy_pkg/loaders/loader_configs/CPI_fred.yaml --outname fred_CPI --clean-dir
    
    - randomized_loader.py
        - Description: Loads standardized datasets, then samples a random subset of the standardized datasets, then selects random subsets of those datasests to form enriched datasets, keeping track of the timestamps for every series. Saves enriched data following the same random sampling scheme until a specified number of enriched datasets are saved.
        - Run Example: python src/synthefy_pkg/loaders/randomized_loader.py --config src/synthefy_pkg/loaders/loader_configs/fred_rand.yaml --outname fred_rand --clean-dir

simple_loader_test.py: unit tests for only the complex filter component of simple_loader.py
