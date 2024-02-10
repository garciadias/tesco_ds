import os

import pandas as pd

from tesco.data.data_types import tesco_dataset_types


def load_raw_data(dataset: str = "tesco_dataset", root_dir="data/") -> pd.DataFrame:
    if dataset in ["tesco_dataset", "masked_dataset"]:
        train_set = pd.read_csv(f"{root_dir}/{dataset}/train.csv")
        if dataset == "tesco_dataset":
            test_set = pd.read_csv(f"{root_dir}/{dataset}/test.csv")
            train_set["is_train"] = True
            test_set["is_train"] = False
            return pd.concat([train_set, test_set], ignore_index=True)
        else:
            return train_set
    else:
        raise ValueError('Invalid dataset name. Please use "tesco_dataset" or "masked_dataset"')


def preprocess_tesco_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df["county"] = df["county"].str.lower().str.replace("c_", "")
    df["transport_availability"] = df["transport_availability"].str.lower().str.replace("transport options", "")
    df["new_store"] = df["new_store"].map({"yes": True, "no": False})
    return df


def load_preprocessed_data(dataset: str = "tesco_dataset", root_dir="data/") -> pd.DataFrame:
    if dataset in ["tesco_dataset", "masked_dataset"]:
        if dataset == "tesco_dataset":
            return pd.read_csv(f"{root_dir}/preprocessed/{dataset}/data.csv", dtype=tesco_dataset_types)
        return pd.read_csv(f"{root_dir}/preprocessed/{dataset}/data.csv")
    raise ValueError('Invalid dataset name. Please use "tesco_dataset" or "masked_dataset"')


def run():
    print("🚀 Starting preprocessing...")
    dataset_list = ["tesco_dataset", "masked_dataset"]
    for dataset_name in dataset_list:
        print(f"\t🗃 Loading {dataset_name} data...")
        df = load_raw_data(dataset_name)
        if dataset_name == "tesco_dataset":
            print(f"\t🔨 Preprocessing {dataset_name} data...")
            df = preprocess_tesco_dataset(df)
        print("\t💾 Saving data...")
        file_path = f"data/preprocessed/{dataset_name}"
        file_name = "data.csv"
        os.makedirs(file_path, exist_ok=True)
        df.to_csv(f"{file_path}/{file_name}", index=False)
        print(f"📦 Data saved at {file_path}/{file_name}")
    print(f"👋 Preprocessing finished for {' and '.join(dataset_list)}")


if __name__ == "__main__":
    run()
