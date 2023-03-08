import pandas as pd

def import_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

def drop_null(data: pd.DataFrame, column: str) -> pd.DataFrame:
    data.drop(data.loc[data[column].isnull()].index, inplace=True)
    return data

def get_imdb_id(data: pd.DataFrame) -> pd.DataFrame:
    imdb_id = [id_.split("/")[-1] for id_ in data["IMDb Link"]]
    data["IMDb id"] = imdb_id
    return data

def netflix_pipeline(path: str, save_path: str) -> pd.DataFrame:
    data = import_data(path)
    data = data.loc[data["Series or Movie"] == "Movie"]

    data.drop(columns=[
        'Title', 'Tags', 'Languages', 'Series or Movie', 'Hidden Gem Score', 'Country Availability', 'Runtime', 
        'Director', 'Writer', 'Actors', 'View Rating', 'Rotten Tomatoes Score', 'Metacritic Score', 'Awards Received',
        'Awards Nominated For', 'Boxoffice', 'Release Date', 'Netflix Release Date', 'Production House',
        'Summary', 'IMDb Votes', 'Image', 'Trailer Site', 'Genre'
    ], inplace=True)

    data = drop_null(data, "IMDb Link")
    data = get_imdb_id(data)

    data.to_csv(save_path)


if __name__ == "__main__":
    netflix_pipeline(
        "./datasets/raw/netflix/Netflix Dataset Latest 2021.xlsx",
        "./datasets/raw/netflix/netflix_processed.csv"
    )