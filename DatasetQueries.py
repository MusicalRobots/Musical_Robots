import pandas as pd
import numpy as np
from typing import List


def return_similar_genres(genre: str, genre_df: pd.DataFrame, track_df: pd.DataFrame, k: int = 10) -> List[str]:
    """
    Return up to k most similar genres to the input genre based on how often genres are reported together,
    ranked by decreasing similarity.

    Args:
        genre: (str) String specifying genre to compare to.
        k: (int) Integer between 1 and 10 specifying up to how many ranked similar genres to return.
        genre_df: (pd.DataFrame) Dataframe storing genre information.
        track_df: (pd.DataFrame) Dataframe storing general track data.

    Outputs:
        most_similar_genres_list: (List[str]) List containing up to k most similar genres to the input genre,
        ranked by decreasing similarity.
    """
    assert k in range(1, 10), "You can only return between 1 and 10 most similar genres."

    genre_id = genre_df[genre_df['title'] == genre]['genre_id']
    co_classified_genres = \
        np.concatenate(track_df[track_df['track_genres']].apply(lambda x: any([genre_id in x]))['track_genres'])

    unique_genres, counts = np.unique(co_classified_genres, return_counts=True)

    indices = np.argsort(counts)[::-1]

    most_similar_genres = unique_genres[indices][1:np.min(len(unique_genres), k)]

    most_similar_genres_list = []

    for genre in most_similar_genres:
        most_similar_genres_list.append([genre_df['genre_id'] == genre]['title'].item())

    return most_similar_genres_list


def return_most_popular_song(genre: str, genre_df: pd.DataFrame, track_df: pd.DataFrame) -> List[str]:
    """
    Return most popular song in a genre according to track listens.  More than one song can be returned if multiple
    songs have the same number of listens.

    Args:
        genre: (str) String specifying genre to search for most popular song.
        genre_df: (pd.DataFrame) Dataframe storing genre information.
        track_df: (pd.DataFrame) Dataframe storing track information.

    Outputs:
        most_popular_songs: (List[str]) Most popular song in a given genre according to track listens.
    """
    genre_id = genre_df[genre_df['title'] == genre]['genre_id']
    tracks_in_genre_df = track_df[track_df['track_genres']].apply(lambda x: any([genre_id in x]))

    most_popular_songs = tracks_in_genre_df[tracks_in_genre_df['track_listens'] ==
                                                max(tracks_in_genre_df['track_listens'])]['track_title'].to_list()

    return most_popular_songs
