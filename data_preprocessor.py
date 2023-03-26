import numpy as np
import pandas as pd


def data_preprocessor(data: pd.DataFrame) -> pd.DataFrame:

    ######## GENRES
    # Which genres are in the dataset
    data_genres = data['genres']
    genres_list = []
    for aux_row in data_genres:
        genres_list.extend(aux_row.replace("'", "").strip("[]").split(", "))

    for genre in np.unique(genres_list):
        data[f"genre_{genre}"] = [int(genre in row.replace("'", "").strip("[]").split(", ")) for row in data_genres]

    genres_list = []
    for aux_row in data_genres:
        try:
            x = aux_row.replace("'", "").strip("[]").split(", ")
        except:
            x = ""
        genres_list.append(", ".join(x))

    data["genres"] = genres_list

    ######## POPULARITY
    classes = [0 if pop < 5 else 1 if pop < 10 else 2 if pop < 15 else 3 if pop < 20 else 4 if pop < 25 else 5 for pop in data["popularity"]]
    data["popularity_class"] = classes  

    ######## ORIGINAL LANGUAGES
    languages = {
        "de":"german", "en":"english", "es":"spanish", "fr":"french", "hi":"hindi", "it":"italian",
        "ja":"japanese", "ko":"korean", "pt":"portuguese", "ru":"russian", "zh":"chinese"
    }
    data["original_language"] = data["original_language"].replace(languages)

    ## One hot encoding on original_languages
    for language in np.unique(data["original_language"]):
        data[f"language_{language}"] = (data["original_language"] == language).values.astype(int)


    ######## PRODUCTION COMPANIES
    # Which companies are in the dataset
    production_companies = data['production_companies']
    p_companies_list = []
    for aux_row in production_companies:
        try:
            x = aux_row.replace("'", "").strip("[]").split(", ")
        except:
            x = ""
        p_companies_list.append(", ".join(x))

    data['production_companies'] = p_companies_list

    ######## PRODUCTION COUNTRIES
    # Which countries are in the dataset
    countries_name = {
        'ar': 'Argentina', 'at': 'Austria', 'au': 'Australia', 'be': 'Belgium', 'br': 'Brazil',
        'ca': 'Canada', 'ch': 'Switzerland', 'cn': 'China', 'de': 'Germany', 'dk': 'Denmark', 'es': 'Spain',
        'fr': 'France', 'gb': 'United Kingdom', 'hk': 'Hong Kong', 'ie': 'Ireland', 'in': 'India',
        'it': 'Italy', 'jp': 'Japan', 'kr': 'Korea', 'mx': 'Mexico', 'nl': 'Netherlands', 'nz': 'New Zealand',
        'pt': 'Portugal', 'ru': 'Russia', 'se': 'Sweden', 'tw': 'Taiwan', 'us': 'United States', 'za': 'South Africa'
    }
    production_countries = data['production_countries']
    p_countries_list = []
    for aux_row in production_countries:
        x = aux_row.replace("'", "").strip("[]").split(", ")
        y = [countries_name[country] for country in x]
        p_countries_list.append(", ".join(y))
        
    data['production_countries'] = p_countries_list

    ######## FINAL VERSIONS
    data_metadata = data[['genres', 'release_year', 'runtime', 'vote_average', 'overview',
        'title', 'original_language', 'popularity', 'production_countries',
        'production_companies', 'belongs_to_collection', 'poster_path',
        'imdb_id', 'tmdb_id']]

    data_model = data[['release_year', 'runtime', 'vote_average',
        'tmdb_id', 'genre_action', 'genre_adventure',
        'genre_animation', 'genre_comedy', 'genre_crime', 'genre_documentary',
        'genre_drama', 'genre_family', 'genre_fantasy', 'genre_foreign',
        'genre_history', 'genre_horror', 'genre_music', 'genre_mystery',
        'genre_romance', 'genre_science_fiction', 'genre_thriller', 'genre_war',
        'genre_western', 'popularity_class', 'language_chinese',
        'language_english', 'language_french', 'language_german',
        'language_hindi', 'language_italian', 'language_japanese',
        'language_korean', 'language_portuguese', 'language_russian',
        'language_spanish']]

    return data, data_metadata, data_model


if __name__ == "__main__":
    dataset_path = "./datasets/movies_dataset_cleaned.csv"

    data = pd.read_csv(dataset_path, index_col=0)

    data, data_metadata, data_model = data_preprocessor(data)

    data.to_csv("./datasets/movies_dataset_full.csv")
    data_metadata.to_csv("./datasets/movies_dataset_metadata.csv")
    data_model.to_csv("./datasets/movies_dataset_model.csv")