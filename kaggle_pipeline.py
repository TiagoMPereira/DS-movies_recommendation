import pandas as pd
import numpy as np

def import_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def list_separator(data: pd.DataFrame, column: str, key_to_keep: str) -> pd.DataFrame:
    values = data.loc[:, column]
    values_list = [
        list_.strip('][').split("}, {") if list_ != "[]" else "[]" for list_ in values
    ]
    values_keys = [
        [
            name.split(f"{key_to_keep}\': \'")[-1].split("\'")[0].lower().replace(" ", "_") for name in row
        ] if row != "[]" else "[]" for row in values_list
    ] 

    data[column] = values_keys
    return data

def get_collection_name(data: pd.DataFrame, column: str) -> pd.DataFrame:
    values = data.loc[:, column]
    collection_name = [
        name.split(f"name\': \'")[-1].split("\'")[0].lower() if name != "[]" else "[]" for name in values
    ]

    data[column] = collection_name
    return data

def kaggle_pipeline(path: str, save_path: str) -> pd.DataFrame:
    data = import_data(path)

    data.drop(columns=['homepage', 'status', 'tagline', 'video', 'vote_count'], inplace=True)

    data["production_companies"].replace(np.nan, "[]", inplace=True)
    data["production_countries"].replace(np.nan, "[]", inplace=True)
    data["spoken_languages"].replace(np.nan, "[]", inplace=True)
    data["belongs_to_collection"].replace(np.nan, "[]", inplace=True)

    data = list_separator(data, "genres", "name")
    data = list_separator(data, "production_companies", "name")
    data = list_separator(data, "production_countries", "iso_3166_1")
    data = list_separator(data, "spoken_languages", "iso_639_1")

    data = get_collection_name(data, "belongs_to_collection")
    data.rename(columns={"id": "tmdb_id"}, inplace=True)

    data.to_csv(save_path)


if __name__ == "__main__":
    kaggle_pipeline(
        "./datasets/raw/movies_kaggle/movies_metadata.csv",
        "./datasets/raw/movies_kaggle/kaggle_processed.csv"
    )