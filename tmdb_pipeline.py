import pandas as pd

def import_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def list_separator(data: pd.DataFrame, column: str, key_to_keep: str) -> pd.DataFrame:
    values = data.loc[:, column]
    values_list = [
        list_.strip('][').split("}, {") for list_ in values
    ]
    values_keys = [
        [
            name.split(f"{key_to_keep}\": \"")[-1].split("\"")[0].lower().replace(" ", "_") for name in row
        ] for row in values_list
    ] 

    data[column] = values_keys
    return data

def tmdb_pipeline(path: str, save_path: str) -> pd.DataFrame:
    data = import_data(path)

    data.drop(columns=['homepage', 'keywords', 'status', 'tagline', 'vote_count'], inplace=True)

    data = list_separator(data, "genres", "name")
    data = list_separator(data, "production_companies", "name")
    data = list_separator(data, "production_countries", "iso_3166_1")
    data = list_separator(data, "spoken_languages", "iso_639_1")

    data.to_csv(save_path)


if __name__ == "__main__":
    tmdb_pipeline(
        "./datasets/raw/tmdb/tmdb_5000_movies.csv",
        "./datasets/raw/tmdb/tmdb_processed.csv"
    )