import numpy as np
import pandas as pd
from collections import Counter

def dropna_row(data: pd.DataFrame, column: str):
    data_drop = data.copy()
    data_drop.drop(data_drop.loc[data_drop[column].isnull()].index, inplace=True)
    data_drop.reset_index(drop=True, inplace=True)
    return data_drop.copy()

def filter_runtime(runtime):
    return 30 <= runtime <= 210

def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:

    data = data[["genres", "release_date", "runtime", "vote_average", "overview",
                 "title", "original_language", "popularity", "production_countries",
                 "production_companies", "belongs_to_collection", "poster_path",
                 "imdb_id", "tmdb_id"]]

    # Indexes
    data.drop_duplicates(subset=["tmdb_id"], keep="first", inplace=True)
    data = dropna_row(data, "imdb_id")

    ####################################################
    # Genres
    ## Which genres are in the dataset
    data_genres = data['genres']
    genres_list = []
    for aux_row in data_genres:
        genres_list.extend(aux_row.replace("'", "").strip("[]").split(", "))

    available_genres = dict(Counter(genres_list))

    ## selecting genres with less than 1000 movies 
    uncommon_genres = [genre for genre, count in available_genres.items() if count < 1000]

    ## Dropping movies with uncommon genres or without genres
    index_to_drop_by_genres = []
    for id_, genres in zip(data_genres.index, data_genres):
        genre = genres.replace("'", "").strip("[]").split(", ")
        has_uncommon = set(genre).intersection(set(uncommon_genres))

        if not genre[0] or has_uncommon:
            index_to_drop_by_genres.append(id_)

    data.drop(index_to_drop_by_genres, inplace=True)
    data.reset_index(drop=True, inplace=True)

    ####################################################
    # Release date
    data = dropna_row(data, "release_date")

    release_year = [int(date.split("-")[0]) for date in data["release_date"]]
    data["release_date"] = release_year
    data.rename(columns={"release_date":"release_year"}, inplace=True)

    ####################################################
    # Runtime
    data = dropna_row(data, "runtime")

    runtime_out_limit = [not(filter_runtime(runtime)) for runtime in data["runtime"]]
    runtime_drop = data.loc[runtime_out_limit].index
    data.drop(runtime_drop, inplace=True)
    data.reset_index(drop=True, inplace=True)

    ####################################################
    # Vote Average
    ## Dropping movies with votes = 0
    low_rated_index = data.loc[data["vote_average"] == 0,:].index
    data.drop(low_rated_index, inplace=True)
    data.reset_index(drop=True, inplace=True)

    ####################################################
    # Overview
    ## Drop null
    empty_overview = data.loc[data["overview"].isnull(),:].index
    data.drop(empty_overview, inplace=True)
    data.reset_index(drop=True, inplace=True)
    ## Drop "No overview found.", "No overview"
    no_overview = ["No overview found.", "No Overview", " "]
    drop_overview = [overview in no_overview for overview in data["overview"]]
    drop_overview_id = data.loc[drop_overview,:].index
    data.drop(drop_overview_id, inplace=True)
    data.reset_index(drop=True, inplace=True)

    ####################################################
    # Original language
    ## keep only the top 10 languages + portuguese
    top_languages = data["original_language"].value_counts().head(10).index.insert(0, "pt")

    data = data.loc[data["original_language"].isin(top_languages)]

    ####################################################
    # Popularity
    data["popularity"] = data["popularity"].astype(float)

    ####################################################
    # Production countries
    ## Which countries are in the dataset
    data_countries = data['production_countries']
    countries_list = []
    for aux_row in data_countries:
        countries_list.extend(aux_row.replace("'", "").strip("[]").split(", "))

    available_countries = dict(Counter(countries_list))

    ## selecting countries with less than 100 movies 
    uncommon_countries = [genre for genre, count in available_countries.items() if count < 100]

    countries_replaced = []
    for id_, countries in zip(data_countries.index, data_countries):
        country = countries.replace("'", "").strip("[]").split(", ")        
        filtered_countries = [c for c in country if c not in uncommon_countries]
        countries_replaced.append(str(filtered_countries))

    data["production_countries"] = countries_replaced
    data.drop(data.loc[data["production_countries"].isin(["['']", "[]"])].index, inplace=True)

    ####################################################
    # Production companies
    data["production_companies"] = data["production_companies"].replace({"[]": np.nan})
  
    ####################################################
    # Bellongs to collection
    data["belongs_to_collection"] = data["belongs_to_collection"].replace({"[]": np.nan, "{": np.nan})


    data.reset_index(drop=True, inplace=True)
    return data


if __name__ == "__main__":
    dataset_path = "./datasets/movies_dataset.csv"
    dataset_save = "./datasets/movies_dataset_cleaned.csv"

    data = pd.read_csv(dataset_path, low_memory=False)

    data_cleaned = data_cleaning(data)

    data_cleaned.to_csv(dataset_save)
