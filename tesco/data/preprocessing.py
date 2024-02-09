import os

import pandas as pd


def load_raw_data(dataset: str = 'tesco_dataset', root_dir="data/") -> pd.DataFrame:
    if dataset in ['tesco_dataset', 'masked_dataset']:
        train_set = pd.read_csv(f'{root_dir}/{dataset}/train.csv')
        if dataset == 'tesco_dataset':
            test_set = pd.read_csv(f'{root_dir}/{dataset}/test.csv', sep=';')
            train_set['is_test'] = False
            test_set['is_test'] = True
            return pd.concat([train_set, test_set], ignore_index=True)
        else:
            return train_set
    else:
        raise ValueError('Invalid dataset name. Please use "tesco_dataset" or "masked_dataset"')


if __name__ == '__main__':
    print("🚀 Starting preprocessing...")
    dataset_list = ['tesco_dataset', 'masked_dataset']
    for dataset_name in dataset_list:
        print(f"\t🗃 Loading {dataset_name} data...")
        df = load_raw_data(dataset_name)
        print("\t💾 Saving data...")
        file_path = f'data/preprocessed/{dataset_name}'
        file_name = 'data.csv'
        os.makedirs(file_path, exist_ok=True)
        df.to_csv(f'{file_path}/{file_name}', index=False)
        print(f"📦 Data saved at {file_path}/{file_name}")
    print(f"👋 Preprocessing finished for {' and '.join(dataset_list)}")
