import pandas as pd
import numpy as np
from typing import Union, List
from sklearn.metrics.pairwise import cosine_similarity


class Scaler(object):

    def __init__(self) -> None:
        pass

    def fit(self, serie: pd.Series) -> None:
        self.max_ = serie.max()
        self.min_ = serie.min()

    def transform(self, value: Union[int, float, pd.Series]) -> Union[pd.Series, float]:
        return np.round((value - self.min_) / (self.max_ - self.min_), 3)

class Languages(object):
    def __init__(self):
        self.chinese = 'language_chinese'
        self.english = 'language_english'
        self.french = 'language_french'
        self.german = 'language_german'
        self.hindi = 'language_hindi'
        self.italian = 'language_italian'
        self.japanese = 'language_japanese'
        self.korean = 'language_korean'
        self.portuguese = 'language_portuguese'
        self.russian = 'language_russian'
        self.spanish = 'language_spanish'

    def none(self):
        pass

class LanguageEncoder(object):

    def __init__(self, languages: Languages) -> None:
        self.languages = languages.__dict__

    def encode(self, value: str) -> dict:
        return {language_name: int(value == language_id)
                for language_id, language_name in self.languages.items()}

class Genres(object):
    def __init__(self):
        self.action = 'genre_action'
        self.adventure = 'genre_adventure'
        self.animation = 'genre_animation'
        self.comedy = 'genre_comedy'
        self.crime = 'genre_crime'
        self.documentary = 'genre_documentary'
        self.drama = 'genre_drama'
        self.family = 'genre_family'
        self.fantasy = 'genre_fantasy'
        self.foreign = 'genre_foreign'
        self.history = 'genre_history'
        self.horror = 'genre_horror'
        self.music = 'genre_music'
        self.mystery = 'genre_mystery'
        self.romance = 'genre_romance'
        self.science_fiction = 'genre_science_fiction'
        self.thriller = 'genre_thriller'
        self.war = 'genre_war'
        self.western = 'genre_western'

    def none(self):
        pass

class GenreEncoder(object):

    def __init__(self, genres: Genres) -> None:
        self.genres = genres.__dict__

    def encode(self, values: List[str]) -> dict:
        return {genre_name: int(genre_id in values)
                for genre_id, genre_name in self.genres.items()}

class PopularityEncoder(object):

    def __init__(self) -> None:
        pass

    def encode(self, value: float) -> int:
        if value < 5:
            return 0
        if value < 10:
            return 1
        if value < 15:
            return 2
        if value < 20:
            return 3
        if value < 25:
            return 4
        return 5
    
class InputPredict(object):
    genre1: str
    genre2: str
    genre3: str
    release_year: int
    runtime: float
    vote_average: float
    popularity: int
    language: str

class MovieModel(object):

    def __init__(self, data: pd.DataFrame, metadata: pd.DataFrame) -> None:
        self.movies = data.set_index("tmdb_id")
        self.metadata = metadata.set_index("tmdb_id")

        self.release_year_scaler = Scaler()
        self.runtime_scaler = Scaler()
        self.vote_average_scaler = Scaler()

        self.languages_encoder = LanguageEncoder(Languages())
        self.genres_encoder = GenreEncoder(Genres())
        self.popularity_encoder = PopularityEncoder()

    def fit(self):
        self.release_year_scaler.fit(self.movies["release_year"])
        self.runtime_scaler.fit(self.movies["runtime"])
        self.vote_average_scaler.fit(self.movies["vote_average"])

        self.movies["release_year"] = self.release_year_scaler.\
            transform(self.movies["release_year"])
        self.movies["runtime"] = self.runtime_scaler.\
            transform(self.movies["runtime"])
        self.movies["vote_average"] = self.vote_average_scaler.\
            transform(self.movies["vote_average"])

    def predict(self, inputPredict: InputPredict, n_predictions=5):

        genres = [inputPredict.genre1, inputPredict.genre2, inputPredict.genre3]
        predict_dict = self.genres_encoder.encode(genres)

        predict_dict.update(self.languages_encoder.encode(inputPredict.language))

        predict_dict.update({"popularity_class": self.popularity_encoder.encode(inputPredict.popularity)})

        predict_dict.update({"release_year": self.release_year_scaler.transform(inputPredict.release_year)})
        predict_dict.update({"runtime": self.runtime_scaler.transform(inputPredict.runtime)})
        predict_dict.update({"vote_average": self.vote_average_scaler.transform(inputPredict.vote_average)})

        predict_df = pd.DataFrame([predict_dict])
        predict_df = predict_df[list(self.movies.columns)]

        X = self.movies.values
        idx = self.movies.index
        y = predict_df.values
        similarity = cosine_similarity(X, y)
        similarity = pd.Series(similarity.reshape(1, -1)[0], idx)
        similarity.sort_values(ascending=False, inplace=True)

        most = similarity.head(n_predictions).index

        recommended = self.metadata.loc[most, :]
        recommended["probability"] = similarity[:n_predictions]
        return recommended




if __name__ == "__main__":
    data = pd.read_csv("./datasets/movies_dataset_model.csv", index_col=0)
    data_metadata = pd.read_csv("./datasets/movies_dataset_metadata.csv", index_col=0)

    model = MovieModel(data, data_metadata)

    model.fit()

    test = InputPredict()
    test.genre1 = "adventure"
    test.genre2 = "family"
    test.genre3 = "fantasy"
    test.release_year = 2001
    test.runtime = 152
    test.vote_average = 7.5
    test.popularity = 38
    test.language = "english"

    recommended = model.predict(test)

    print(recommended)