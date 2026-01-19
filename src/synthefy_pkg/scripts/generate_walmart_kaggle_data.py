import kagglehub
import shutil
import os
import pandas as pd
import numpy as np
import random
import argparse
import json
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Walmart Kaggle dataset with customizable parameters')
    parser.add_argument('--window-size', type=int, default=64,
                        help='Size of the sliding window (default: 64)')
    parser.add_argument('--stride', type=int, default=4,
                        help='Stride size for the sliding window (default: 4)')
    parser.add_argument('--use-custom-split', action='store_true',
                        help='Use custom train/val/test split (default: False)')
    parser.add_argument('--no-use-store', action='store_true', default=False,
                        help='Not use store as a feature (default: False)')
    parser.add_argument('--no-use-dept', action='store_true', default=False,
                        help='Not use department as a feature (default: False)')
    parser.add_argument('--no-features', action='store_true', default=False,
                        help='Not use features (default: False)')
    parser.add_argument('--no-shuffle', action='store_true', default=False,
                        help='Not use shuffling (default: False)')
    parser.add_argument('--num-discrete-conditions', type=int, default=125,
                        help='Number of discrete conditions (default: 125)')

    args = parser.parse_args()
    # Download latest version
    path = kagglehub.dataset_download("aslanahmedov/walmart-sales-forecast")

    print("downloaded dataset files:", path, os.listdir(path))


    # Replace the hardcoded variables with command line arguments
    WINDOW_SIZE = args.window_size
    STRIDE = args.stride
    USE_CUSTOM_SPLIT = args.use_custom_split
    USE_STORE = not args.no_use_store
    USE_DEPT = not args.no_use_dept
    NO_FEATURES = args.no_features
    SHUFFLE = not args.no_shuffle
    NUM_DISCRETE_CONDITIONS = args.num_discrete_conditions

    PREPROCESS_NAME = "custom" if USE_CUSTOM_SPLIT else "not_custom"
    PREPROCESS_NAME += f"_w{WINDOW_SIZE}s{STRIDE}"
    PREPROCESS_NAME += "_store_dept" if USE_STORE and USE_DEPT else ("_dept" if USE_DEPT else "_store")
    PREPROCESS_NAME += "_no_features" if NO_FEATURES else ""
    PREPROCESS_NAME += "_no_shuffle" if not SHUFFLE else ""
    # move files to this folder
    file_names = os.listdir(path)

    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "synthefy_data", "walmart_sales")

    new_folder = data_folder.rstrip("/") + "_" + PREPROCESS_NAME
    os.makedirs(new_folder, exist_ok=True)

    for file_name in file_names:
        print(os.path.join(path, file_name), os.path.join(new_folder, file_name))
        shutil.move(os.path.join(path, file_name), os.path.join(new_folder, file_name))
    os.rmdir(path)

    print("moved to dataframe folder", new_folder)

    # construct formatted CSV by adding in the features from features.csv
    train_data = pd.read_csv(os.path.join(new_folder, "train.csv"))
    features = pd.read_csv(os.path.join(new_folder, "features.csv"))
    store_data = pd.read_csv(os.path.join(new_folder, "stores.csv"))

    # swap train features
    if USE_STORE and USE_DEPT:  train_data = train_data[["Date","Store","Dept","Weekly_Sales","IsHoliday"]]
    elif USE_DEPT: train_data = train_data[["Date","Dept","Weekly_Sales","IsHoliday","Store"]]
    elif USE_STORE: train_data = train_data[["Date","Store","Weekly_Sales","IsHoliday","Dept"]]
    else: raise ValueError("No store or dept features selected")

    print("\nNumber of unique Store-Department combinations:", len(train_data[['Store', 'Dept']].drop_duplicates()))

    # features should correspond to the date and store, and be replicated in other positions
    df = pd.merge_ordered(train_data, store_data, fill_method="ffill", on=["Store"], how="inner")
    if not NO_FEATURES: df = pd.merge_ordered(df, features, fill_method="ffill", on=["Store", "Date"], how="inner")
    df["Date"] = pd.to_datetime(df["Date"])

    # treat store and dept as a group variables
    if USE_STORE and USE_DEPT: df = df.sort_values(by=["Store", "Dept", "Date"])
    elif USE_DEPT: df = df.sort_values(by=["Dept", "Date"])
    elif USE_STORE: df = df.sort_values(by=["Store", "Date"])
    else: raise ValueError("No store or dept features selected")

    df = df.drop('IsHoliday_y', axis=1)
    df.rename(columns={'IsHoliday_x': 'IsHoliday'}, inplace=True)

    # add a "custom split" column for splitting stores into train, validation or test
    # this is necessary because the stratified sampling will force window sizes to be small
    if USE_CUSTOM_SPLIT:
        TRAIN_SPLIT = 0.8
        VAL_SPLIT = 0.1
        store_depts = np.array(list(df[['Store', 'Dept']].drop_duplicates().itertuples(index=False, name=None)))
        train_idxes = set(random.sample(range(len(store_depts)), int(len(store_depts) * TRAIN_SPLIT)))
        val_idxes = [i for i in range(len(store_depts)) if i not in train_idxes]
        val_idxes = random.sample(val_idxes, int(len(store_depts) * VAL_SPLIT))
        train_sds = set([(sd[0], sd[1]) for sd in store_depts[list(train_idxes)]])
        val_sds = set([(sd[0], sd[1]) for sd in store_depts[list(val_idxes)]])
        df["custom_split"] = df.apply(lambda row: 
                                    ("train" if (row["Store"], row["Dept"]) in train_sds else (
                                    "val" if (row["Store"], row["Dept"]) in val_sds else (
                                    "test"
                                    ))), axis=1)



    # print out values to check correctness
    with pd.option_context('display.max_rows', None,
                        'display.max_columns', None,
                        'display.precision', 3,
                        ):
        print("data shapes", df.shape, train_data.shape)
        print(df.head(10))
        print(df.sample(10))



    # merge test data also
    print("*****************Merging Test data****************")
    test_data = pd.read_csv(os.path.join(new_folder, "test.csv"))
    if USE_STORE and USE_DEPT: test_data = test_data[["Date","Store","Dept","IsHoliday"]]
    elif USE_DEPT: test_data = test_data[["Date","Dept","IsHoliday","Store"]]
    elif USE_STORE: test_data = test_data[["Date","Store","IsHoliday","Dept"]]
    else: raise ValueError("No store or dept features selected")


    test_df = pd.merge_ordered(test_data, store_data, fill_method="ffill", on=["Store"], how="inner")
    if not NO_FEATURES: test_df = pd.merge_ordered(test_df, features, fill_method="ffill", on=["Store", "Date"], how="inner")

    if USE_STORE and USE_DEPT: test_df = test_df.sort_values(by=["Store", "Dept", "Date"])
    elif USE_DEPT: test_df = test_df.sort_values(by=["Dept", "Date"])
    elif USE_STORE: test_df = test_df.sort_values(by=["Store", "Date"])
    else: raise ValueError("No store or dept features selected")


    test_df = test_df.drop('IsHoliday_y', axis=1)
    test_df.rename(columns={'IsHoliday_x': 'IsHoliday'}, inplace=True)

    # printouts to check values
    with pd.option_context('display.max_rows', None,
                        'display.max_columns', None,
                        'display.precision', 3,
                        ):
        print("data shapes", test_df.shape, test_data.shape)
        print(test_df.head(10))
        print(test_df.sample(10))


    # write to parquet
    df.to_parquet(os.path.join(new_folder, 'merged_features_train.parquet'), engine='pyarrow')
    test_df.to_parquet(os.path.join(new_folder, 'merged_features_test.parquet'), engine='pyarrow')

    # Write preprocessing config JSON file using the desired values
    config_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "examples", "configs", "preprocessing_configs")
    with open(os.path.join(config_folder, "config_walmart_sales_preprocessing.json"), "r") as f:
        config = json.load(f)
    
    # Automatically set window size and stride  
    config["window_size"] = WINDOW_SIZE
    config["stride"] = STRIDE
    config["shuffle"] = SHUFFLE
    config["filename"] = f"walmart_sales_{PREPROCESS_NAME}/merged_features_train.parquet"

    if NO_FEATURES: 
        config["continuous"]["cols"] = []
        config["discrete"]["cols"] = ["IsHoliday"]
    # Automatically set group labels and discrete features (default is store and dept)
    if not (USE_STORE and USE_DEPT): 
        if USE_STORE: 
            config["group_labels"]["cols"] = ["Store"]
            config["discrete"]["cols"] = ["Dept"] + config["discrete"]["cols"]
        elif USE_DEPT: 
            config["group_labels"]["cols"] = ["Dept"]
            config["discrete"]["cols"] = ["Store"] + config["discrete"]["cols"]
    with open(os.path.join(config_folder, f"config_walmart_sales_{PREPROCESS_NAME}_preprocessing.json"), "w") as f:
        json.dump(config, f, indent=4)


    # Write synthesis config JSON file using the desired values
    config_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "examples", "configs", "synthesis_configs")
    with open(os.path.join(config_folder, "config_walmart_sales_synthesis.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    # Automatically set window size and stride  
    config["dataset_config"]["time_series_length"] = WINDOW_SIZE
    config["execution_config"]["run_name"] = f"synthefy_forecasting_model_v1_walmart_sales_{PREPROCESS_NAME}"
    config["dataset_config"]["dataset_name"] = f"walmart_sales_{PREPROCESS_NAME}"

    # Automatically set group labels and discrete features (default is store and dept)
    config["dataset_config"]["num_discrete_conditions"] = NUM_DISCRETE_CONDITIONS

    if not (USE_STORE and USE_DEPT): 
        config["dataset_config"]["num_discrete_labels"] = 4
    with open(os.path.join(config_folder, f"config_walmart_sales_{PREPROCESS_NAME}_synthesis.yaml"), "w") as f:
        yaml.dump(config, f)
    
    # Write forecasting config JSON file using the desired values
    config_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "examples", "configs", "forecast_configs")
    with open(os.path.join(config_folder, "config_walmart_sales_forecasting.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    # Automatically set window size and stride  
    config["dataset_config"]["time_series_length"] = WINDOW_SIZE
    config["execution_config"]["run_name"] = f"synthefy_forecasting_model_v1_walmart_sales_{PREPROCESS_NAME}"
    config["dataset_config"]["dataset_name"] = f"walmart_sales_{PREPROCESS_NAME}"
    config["dataset_config"]["num_discrete_conditions"] = NUM_DISCRETE_CONDITIONS

    # Austomatically set group labels and discrete features (default is store and dept)
    if not (USE_STORE and USE_DEPT): 
        config["dataset_config"]["num_discrete_labels"] = 4
    with open(os.path.join(config_folder, f"config_walmart_sales_{PREPROCESS_NAME}_forecasting.yaml"), "w") as f:
        yaml.dump(config, f)

    models_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "synthefy_data", "models", PREPROCESS_NAME)
    os.makedirs(models_folder, exist_ok=True)

    # Write TSTR config JSON file for walmart data
    config_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "examples", "configs", "synthefy_configs")
    tstr_config_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "examples", "configs", "tstr_configs")
    generation_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "synthefy_data", "generation_logs", f"walmart_sales_{PREPROCESS_NAME}", "Time_Series_Diffusion_Training", f"synthefy_forecasting_model_v1_walmart_sales_{PREPROCESS_NAME}")
    train_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "synthefy_data", "train_logs", f"synthefy_forecasting_model_v1_walmart_sales_{PREPROCESS_NAME}")
    def create_tstr_config(train_form="tstr"):
        with open(os.path.join(config_folder, "config_walmart_sales_synthesis.yaml"), "r") as f:
            config = yaml.safe_load(f)
        
        # Automatically set window size and stride  
        config["dataset_config"]["time_series_length"] = WINDOW_SIZE
        config["execution_config"]["run_name"] = f"synthefy_forecasting_model_v1_walmart_sales_{PREPROCESS_NAME}"
        config["dataset_config"]["dataset_name"] = f"walmart_sales_{PREPROCESS_NAME}"
        config["dataset_config"]["num_discrete_conditions"] = NUM_DISCRETE_CONDITIONS
        config["tstr_config"]["dataset"]["required_time_series_length"] = WINDOW_SIZE
        config["tstr_config"]["dataset"]["time_series_length"] = WINDOW_SIZE
        config["tstr_config"]["training"]["log_dir"] = os.path.join(train_folder, train_form)

        if not (USE_STORE and USE_DEPT): 
            config["dataset_config"]["num_discrete_labels"] = 4
        config["tstr_config"]["train_dataset_paths"] = [{"path": os.path.join(generation_folder, "train_dataset"), "synthetic_or_original": "synthetic"},
                                                        {"path": os.path.join(generation_folder, "train_dataset"), "synthetic_or_original": "original"},
                                                        {"path": os.path.join(generation_folder, "train_dataset"), "synthetic_or_original": "synthetic"},
                                                        ]
        config["tstr_config"]["test_dataset_paths"] = [{"path": os.path.join(generation_folder, "test_dataset"), "synthetic_or_original": "original"},
                                                        ]
        config["tstr_config"]["val_dataset_paths"] = [{"path": os.path.join(generation_folder, "val_dataset"), "synthetic_or_original": "synthetic"},
                                                        {"path": os.path.join(generation_folder, "val_dataset"), "synthetic_or_original": "original"},
                                                        ]
        return config
    
    train_forms = ["tstr", "trtr", "trstr"]
    for train_form in train_forms:
        config = create_tstr_config(train_form)
        if train_form == "tstr":
            config["tstr_config"]["synthetic_or_original_or_custom"] = "synthetic"
        elif train_form == "trtr":
            config["tstr_config"]["synthetic_or_original_or_custom"] = "original"
        elif train_form == "trstr":
            config["tstr_config"]["synthetic_or_original_or_custom"] = "custom"
        os.makedirs(tstr_config_folder, exist_ok=True)   
        with open(os.path.join(tstr_config_folder, f"config_walmart_sales_{PREPROCESS_NAME}_{train_form}.yaml"), "w") as f:
            yaml.dump(config, f)

    # # Write TRTR config JSON file using the desired values
    # config = create_tstr_config()
    # config["tstr_config"]["synthetic_or_original_or_custom"] = "original"
    # with open(os.path.join(config_folder, f"config_walmart_sales_{PREPROCESS_NAME}_trtr.yaml"), "w") as f:
    #     yaml.dump(config, f)

    # # Write TRSTR config JSON file using the desired values
    # config = create_tstr_config()
    # config["tstr_config"]["synthetic_or_original_or_custom"] = "custom"
    # with open(os.path.join(config_folder, f"config_walmart_sales_{PREPROCESS_NAME}_trstr.yaml"), "w") as f:
    #     yaml.dump(config, f)

