"""Create dataset queries."""

from typing import List, Optional, Tuple
import random
import pandas as pd
import numpy as np
import librosa
import IPython.display as ipd
import audioread


def return_similar_genres(
    genre: str, genre_df: pd.DataFrame, track_df: pd.DataFrame, k: int = 10
) -> List[str]:
    """
    Return up to k most similar genres to the input genre.

    Ranking is based on how often genres are reported together,ranked by
    decreasing similarity.

    Args:
        genre: (str) String specifying genre to compare to.
        k: (int) Integer between 1 and 10 specifying up to how many
        ranked similar genres to return.
        genre_df: (pd.DataFrame) Dataframe storing genre information.
        track_df: (pd.DataFrame) Dataframe storing general track data.

    Outputs:
        most_similar_genres_list: (List[str]) List containing up to k
        most similar genres to the input genre, ranked by decreasing
        similarity.
    """
    assert k in range(
        1, 11
    ), "You can only return between 1 and 10 most similar genres."

    genre_id = genre_df[
        genre_df["title"].apply(
            lambda x: x.lower().replace("-", "").replace(" ", "")
            == genre.lower().replace("-", "").replace(" ", "")
        )
    ]["genre_id"]

    if len(genre_id) == 0:
        raise RuntimeError("Genre does not exist in dataset.")

    genre_id = genre_id.item()

    co_classified_genres = np.concatenate(
        track_df[track_df["track_genres"].apply(
            lambda x: any([genre_id in x]))][
            "track_genres"
        ].to_list()
    )

    unique_genres, counts = np.unique(co_classified_genres, return_counts=True)

    indices = np.argsort(counts)[::-1]

    most_similar_genres = unique_genres[indices][
                          1: min(len(unique_genres), k + 1)]

    most_similar_genres_list = []

    for similar_genre in most_similar_genres:
        most_similar_genres_list.append(
            genre_df[genre_df["genre_id"] == similar_genre]["title"].item()
        )

    return most_similar_genres_list


def return_most_popular_song(
    genre: str, genre_df: pd.DataFrame, track_df: pd.DataFrame
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Return most popular song in a genre according to track listens.

    More than one song can be returned if songs have the same number
    of listens.

    Args:
        genre: (str) String specifying genre to search for most popular song.
        genre_df: (pd.DataFrame) Dataframe storing genre information.
        track_df: (pd.DataFrame) Dataframe storing track information.

    Outputs:
        most_popular_songs: (List[str]) Most popular song in a given genre
        according to track listens.
    """
    genre_id = genre_df[
        genre_df["title"].apply(
            lambda x: x.lower().replace("-", "").replace(" ", "")
            == genre.lower().replace("-", "").replace(" ", "")
        )
    ]["genre_id"]

    if len(genre_id) == 0:
        raise RuntimeError("Genre does not exist in dataset.")

    genre_id = genre_id.item()

    tracks_in_genre_df = track_df[
        track_df["track_genres"].apply(lambda x: any([genre_id in x]))
    ]

    songs_info = tracks_in_genre_df[
        tracks_in_genre_df["track_listens"] == max(
            tracks_in_genre_df["track_listens"])
    ][["track_id", "track_title", "artist_name", "album_title"]]

    most_popular_song_ids = songs_info["track_id"].to_list()
    most_popular_songs = songs_info["track_title"].to_list()
    artists = songs_info["artist_name"].to_list()
    albums = songs_info["album_title"].to_list()

    return most_popular_song_ids, most_popular_songs, artists, albums


def play_random_song_from_genre(
    genre: str, genre_df: pd.DataFrame, track_df: pd.DataFrame,
        path_df: pd.DataFrame
) -> Optional[Tuple[ipd.Audio, str, str, str]]:
    """
    Play a random song from a given genre.

    Args:
        genre: (str) String specifying genre to play a random song from.
        genre_df: (pd.DataFrame) Dataframe storing genre information.
        track_df: (pd.DataFrame) Dataframe storing track information.
        path_df: (pd.DataFrame) Dataframe storing path information.

    Returns:
        (ipd.Audio): Interactive playable file of a random song
        from the specified genre.
    """
    audio = None
    song_title = "NA"
    artist_name = "NA"
    album_title = "NA"

    genre_id = genre_df[
        genre_df["title"].apply(
            lambda x: x.lower().replace("-", "").replace(" ", "")
            == genre.lower().replace("-", "").replace(" ", "")
        )
    ]["genre_id"]

    if len(genre_id) == 0:
        raise RuntimeError("Genre does not exist in dataset")

    genre_id = genre_id.item()

    genre_tracks = track_df[
        track_df["track_genres"].apply(lambda x: any([genre_id in x]))
    ]

    song_ids = genre_tracks["track_id"].to_list()

    status = False
    max_lookups = 0

    while status is False:
        rand_int = random.randint(0, len(song_ids))
        song_id = song_ids[rand_int]
        song_id = str(song_id).zfill(6)

        file_path_genre = path_df[path_df["file_path"].str.contains(song_id)]

        if len(file_path_genre) != 0:
            filename = "data/fma_small/" + file_path_genre["file_path"].item()

            try:
                y_audio, sample_rate = librosa.load(
                    filename, sr=None, mono=True)
                audio = ipd.Audio(data=y_audio, rate=sample_rate)
            except (RuntimeError, audioread.NoBackendError):
                continue

            song_info = track_df[track_df["track_id"] == song_ids[rand_int]][
                ["track_title", "artist_name", "album_title"]
            ]
            song_title = song_info["track_title"].item()
            artist_name = song_info["artist_name"].item()
            album_title = song_info["album_title"].item()

            status = True

        if max_lookups > 10:
            print("Could not find song in genre.")
            return None

        max_lookups += 1

    return audio, song_title, artist_name, album_title


def play_song_from_title(
    title: str, track_df: pd.DataFrame, path_df: pd.DataFrame
) -> Optional[ipd.Audio]:
    """
    Play a song from its title.

    Args:
        title: (str) Title of track to play audio.
        track_df: (pd.DataFrame) Dataframe containing track information.
        path_df: (pd.DataFrame) Dataframe containing path information)

    Outputs:
        (Optional[ipd.Audio]): Interactive playable file of song by
        the specified title.
    """
    song_ids = track_df[track_df["track_title"] == title]

    song_ids.reset_index(inplace=True, drop=True)

    row_number = 0
    audio = None

    if len(song_ids) == 0:
        raise RuntimeError(
            "Song by the title of ", title, " does not exist in dataset."
        )
    if len(song_ids) > 1:
        print(
            "There are multiple songs with this title."
            "Which would you like to hear? Return the corresponding "
            "row number."
        )

        ipd.display(song_ids[["album_title", "artist_name", "track_title"]])
        row_number = input("Enter the row number: ")
        row_number = int(row_number)

        if (row_number > len(song_ids) - 1) or (row_number < 0):
            row_number = input("Invalid row number.  Enter the row number:")
            row_number = int(row_number)

    song_id = song_ids["track_id"].to_list()[row_number]

    song_id = str(song_id).zfill(6)

    file_path = path_df[path_df["file_path"].str.contains(song_id)]

    if len(file_path) != 0:
        filename = "data/fma_small/" + file_path["file_path"].item()
        audio = play_song_from_filename(filename=filename)
    else:
        print("Audio file could not be found.")

    return audio


def play_song_from_filename(filename: str) -> Optional[ipd.Audio]:
    """
    Play a song from its' filename.

    Args:
        filename: (str) Filename to read audio from.
    Returns:
        (ipd.Audio) Audio clip.
    """
    try:
        y_audio, sample_rate = librosa.load(filename, sr=None, mono=True)
        audio = ipd.Audio(data=y_audio, rate=sample_rate)
    except (RuntimeError, audioread.NoBackendError):
        print("Error loading audio.")
        audio = None
    return audio