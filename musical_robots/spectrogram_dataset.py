"""Create a custom dataset that turns mp3 files into spectrograms."""

import warnings
import ast
from typing import Tuple
import audioread
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


class AudioFeature:
    """Create custom dataset of audio features and genre labels."""

    def __init__(
        self, path_to_data="data/fma_small/",
        path_df: pd.DataFrame = None,
        music_df: pd.DataFrame = None,
        genre_df: pd.DataFrame = None,
    ) -> None:
        """
        Create list of data tuples.

        Args:
            path_to_data: (str)
            path_df: (pd.Dataframe) DataFrame containing file paths.
            music_df: (pd.DataFrame) Dataframe containing music information.
            genre_df: (pd.DataFrame) Dataframe containing genre information.

        Attributes:
            self.samples:(List[Tuple[librosa.feature.melspectrogram, str]])
            List of data tuples.
        """
        self.samples = []

        for row in path_df.itertuples():
            filename = path_to_data + row[1]

            try:
                y_audio, sample_rate = librosa.load(
                    filename, sr=None, mono=True)
            except (RuntimeError, audioread.NoBackendError):
                print("Failed to load ", filename)
                continue
            except FileNotFoundError:
                print('File not found', filename)
                continue

            mel = librosa.feature.mfcc(y=y_audio, sr=sample_rate)
            mel = (mel - np.min(mel)) / (np.max(mel) - np.min(mel))

            zero_crossing_rate = librosa.feature.zero_crossing_rate(
                y_audio, frame_length=2048, hop_length=1024
            )
            spectral_centroid = librosa.feature.spectral_centroid(
                y_audio, sr=sample_rate, n_fft=2048, hop_length=1024
            )
            spectral_contrast = librosa.feature.spectral_contrast(
                y_audio, sr=sample_rate, n_fft=2048, hop_length=1024
            )
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y_audio, sr=sample_rate, n_fft=2048, hop_length=1024
            )
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y_audio, sr=sample_rate, n_fft=2048, hop_length=1024
            )

            song_id = row[1].rsplit("/")[1].rsplit(".")[0].lstrip("0")
            try:
                genre_text = music_df[music_df["track_id"] == int(song_id)][
                    "track_genre_top"
                ].item()
            except ValueError:
                continue

            genre_label = genre_df[genre_df[
                                       "title"] == genre_text].index.item()

            self.samples.append(
                (
                    mel,
                    zero_crossing_rate,
                    spectral_centroid,
                    spectral_contrast,
                    spectral_bandwidth,
                    spectral_rolloff,
                    genre_label,
                )
            )

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.samples)

    def __getitem__(
        self, idx
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Return a single audio feature instance and its label from the dataset.

        Args:
            idx: (int) Index of dataset item to return.
        """
        return self.samples[idx]


def create_dataframes(
    file_paths_path: str, tracks_csv_path: str, genre_csv_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create dataframes for audio analysis.

    Args:
        file_paths_path: (str) Path to 'all_data_paths.txt' storing
        the data paths for each sound file.
        tracks_csv_path: (str) Path to 'tracks.csv' containing
        general track data.
        genre_csv_path: (str) Path to 'genre.csv' containing genre information.

    Returns:
        file_path_df: (pd.DataFrame) Dataframe storing the data paths
        for each sound file.
        track_df: (pd.DataFrame) Dataframe storing general track data.
        relevant_genre_df: (pd.DataFrame) Dataframe storing genre information.
        genre_df: (pd.DataFrame) Dataframe storing genre information for
        all genres in the track_df.
    """
    file_path_df = pd.read_csv(file_paths_path,
                               header=None, names=["file_path"])

    track_df = pd.read_csv(
        tracks_csv_path,
        skiprows=[0, 1, 2],
        usecols=[0, 6, 8, 11, 26, 39, 40, 41, 44, 47, 52],
        names=[
            "track_id",
            "album_id",
            "album_listens",
            "album_title",
            "artist_name",
            "track_favorites",
            "track_genre_top",
            "track_genres",
            "track_interest",
            "track_listens",
            "track_title",
        ],
    )

    track_df = track_df[track_df["track_genre_top"].notna()]

    def row_filter(row):
        return [int(i) for i in ast.literal_eval(row["track_genres"])]

    track_df["track_genres"] = track_df.apply(row_filter, axis=1)

    genre_df = pd.read_csv(genre_csv_path, usecols=[0, 3])
    relevant_indices = []
    for genre in track_df["track_genre_top"].unique():
        relevant_indices.append(
            genre_df[genre_df["title"] == genre].index.item())
    relevant_genre_df = genre_df.iloc[relevant_indices]
    relevant_genre_df.reset_index(inplace=True)

    return file_path_df, track_df, relevant_genre_df, genre_df


def split_data(
    file_path_df: pd.DataFrame, test_percentage: float,
        validation_percentage: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into 3 datasets for training, testing and validation.

    Args:
        file_path_df: (pd.DataFrame) Dataframe storing the data paths
        for each sound file.
        test_percentage: (float) Percentage of paths to designate as part
         of the test dataset.
        validation_percentage: (float) Percentage of paths to designate
        as part of the validation dataset.

    Returns:
        train_df: (pd.DataFrame) DataFrame of data for the training set
        test_df: (pd.DataFrame) Dataframe of data for the testing set
        validation_df: (pd.DataFrame) Dataframe of data for the validation set
    """
    train_df, test_df = train_test_split(
        file_path_df, test_size=test_percentage)

    train_df, validation_df = train_test_split(
        train_df, test_size=validation_percentage
    )
    return train_df, test_df, validation_df


def create_audio_feature_dataset(
    file_path_df: pd.DataFrame,
    track_df: pd.DataFrame,
    genre_df: pd.DataFrame,
    path_to_data: str = "data/fma_small/",
    test_percentage: float = 0.10,
    validation_percentage: float = 0.10,
) -> Tuple[AudioFeature, AudioFeature, AudioFeature]:
    """
    Create the custom dataset given the locations of the data.

    Args:
        file_path_df: (pd.DataFrame) Dataframe storing the data paths
        for each sound file.
        track_df: (pd.DataFrame) Dataframe storing general track data.
        genre_df: (pd.Dataframe) Dataframe storing genre information.
        path_to_data: (str) Path to where audio data is stored.
        test_percentage: (float) Percentage of paths to designate as
        part of the test dataset.
        validation_percentage: (float) Percentage of paths to designate
        as part of the validation dataset.

    Returns:
        train_data: (AudioFeature) Dataset containing training music data as
        spectrograms and genre labels.
        validation_data: (AudioFeature) Dataset containing validation music
        data as spectrograms and genre labels.
        test_data: (AudioFeature) Dataset containing testing music data as
        spectrograms and genre labels.
    """

    train_df, test_df, validation_df = split_data(
        file_path_df=file_path_df,
        test_percentage=test_percentage,
        validation_percentage=validation_percentage,
    )

    print("Starting making train_data")
    train_data = AudioFeature(
        path_to_data=path_to_data,
        path_df=train_df,
        music_df=track_df,
        genre_df=genre_df,
    )

    print("Training dataset created.")

    validation_data = AudioFeature(
        path_to_data=path_to_data,
        path_df=validation_df,
        music_df=track_df,
        genre_df=genre_df
    )

    print("Validation dataset created.")

    test_data = AudioFeature(
        path_to_data=path_to_data,
        path_df=test_df,
        music_df=track_df,
        genre_df=genre_df
    )

    print("Test dataset created.")

    return train_data, validation_data, test_data
