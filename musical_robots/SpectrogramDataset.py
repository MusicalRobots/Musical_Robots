"""Create a custom dataset that turns mp3 files into spectrograms."""
import audioread
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import warnings
import librosa
import ast
from typing import Tuple
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


class SpectrogramDataset(Dataset):
    """Create custom dataset of spectrograms and genre labels."""

    def __init__(self, path_df: pd.DataFrame, music_df: pd.DataFrame, genre_df: pd.DataFrame, limit_samples=False, max_samples=100, samples=None) -> None:
        """
        Create list of data tuples.

        Args:
            path_df: (pd.Dataframe) DataFrame containing file paths.
            music_df: (pd.DataFrame) Dataframe containing music information.
            genre_df: (pd.DataFrame) Dataframe containing genre information.

        Attributes:
            self.samples: (List[Tuple[librosa.feature.melspectrogram, str]]) List of data tuples.
        """
        if samples is not None:
            self.samples = samples
        else:
        # self.data = []
        # self.genres = []

            print("len of df is ", len(path_df.index))
            for row in tqdm(path_df.itertuples()):
                if limit_samples:
                    if len(self.samples) > max_samples:
                        break
                filename = 'data/fma_small/' + row[1]

                try:
                    y, sr = librosa.load(filename, sr=None, mono=True)
                except (RuntimeError, audioread.NoBackendError):
                    print('Failed to load ', filename)
                    continue

                genre_label = np.zeros(genre_df['genre_id'].nunique())

                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
                mel = torch.from_numpy(mel)
                m = torch.nn.AdaptiveAvgPool1d(1292)
                mel = m(mel)
                mel = (mel - torch.min(mel))/(torch.max(mel) - torch.min(mel))

                song_id = row[1].rsplit('/')[1].rsplit('.')[0].lstrip('0')

            index = genre_df[genre_df['title'] == genre_text].index.item()
            genre_label[index] = 1

            genre_label = torch.from_numpy(genre_label)

                genre_label = torch.from_numpy(genre_label)

                self.samples.append((mel, genre_label))

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a single spectrogram and its label from the dataset.

        Args:
            idx: (int) Index of dataset item to return.
        """
        return self.samples[idx]


class MfccDataset(Dataset):
    """Create custom dataset of spectrograms and genre labels."""

    def __init__(self, path_df: pd.DataFrame, music_df: pd.DataFrame, genre_df: pd.DataFrame) -> None:
        """
        Create list of data tuples.

        Args:
            path_df: (pd.Dataframe) DataFrame containing file paths.
            music_df: (pd.DataFrame) Dataframe containing music information.
            genre_df: (pd.DataFrame) Dataframe containing genre information.

        Attributes:
            self.samples: (List[Tuple[librosa.feature.melspectrogram, str]]) List of data tuples.
        """
        self.samples = []

        for row in path_df.itertuples():
            filename = 'data/fma_small/' + row[1]

            try:
                y, _ = librosa.load(filename, sr=None, mono=True)
            except (RuntimeError, audioread.NoBackendError):
                print('Failed to load ', filename)
                continue

            genre_label = np.zeros(genre_df['genre_id'].nunique())

            mel = librosa.feature.mfcc(y=y)
            mel = torch.from_numpy(mel)
            mel = (mel - torch.min(mel)) / (torch.max(mel) - torch.min(mel))

            song_id = row[1].rsplit('/')[1].rsplit('.')[0].lstrip('0')
            try:
                genre_text = music_df[music_df['track_id'] == int(song_id)]['track_genre_top'].item()
            except ValueError:
                continue

            index = genre_df[genre_df['title'] == genre_text].index.item()

            genre_label[index] = 1

            genre_label = torch.from_numpy(genre_label)

            self.samples.append((mel, genre_label))

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a single spectrogram and its label from the dataset.

        Args:
            idx: (int) Index of dataset item to return.
        """
        return self.samples[idx]


class AudioFeature(Dataset):
    """Create custom dataset of spectrograms and genre labels."""

    def __init__(self, path_df: pd.DataFrame = None, music_df: pd.DataFrame = None,
                genre_df: pd.DataFrame = None, limit_samples=False,
                max_samples=100) -> None:
        """
        Create list of data tuples.

        Args:
            path_df: (pd.Dataframe) DataFrame containing file paths.
            music_df: (pd.DataFrame) Dataframe containing music information.
            genre_df: (pd.DataFrame) Dataframe containing genre information.

        Attributes:
            self.samples: (List[Tuple[librosa.feature.melspectrogram, str]]) List of data tuples.
        """
        self.samples = []

        for row in path_df.itertuples():
            filename = 'data/fma_small/' + row[1]

            try:
                y, sr = librosa.load(filename, sr=None, mono=True)
            except (RuntimeError, audioread.NoBackendError):
                print('Failed to load ', filename)
                continue

            mel = librosa.feature.mfcc(y=y, sr=sr)
            mel = (mel - np.min(mel)) / (np.max(mel) - np.min(mel))

            zero_crossing_rate = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=1024)
            spectral_centroid = librosa.feature.spectral_centroid(y, sr=sr, n_fft=2048, hop_length=1024)
            spectral_contrast = librosa.feature.spectral_contrast(y, sr=sr, n_fft=2048, hop_length=1024)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y, sr=sr, n_fft=2048, hop_length=1024)
            spectral_rolloff = librosa.feature.spectral_rolloff(y, sr=sr, n_fft=2048, hop_length=1024)

            song_id = row[1].rsplit('/')[1].rsplit('.')[0].lstrip('0')
            try:
                genre_text = music_df[music_df['track_id'] == int(song_id)]['track_genre_top'].item()
            except ValueError:
                continue

            genre_label = genre_df[genre_df['title'] == genre_text].index.item()

            self.samples.append((mel, zero_crossing_rate, spectral_centroid, spectral_contrast,
                                 spectral_bandwidth, spectral_rolloff, genre_label))

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                        np.ndarray]:
        """
        Return a single spectrogram and its label from the dataset.

        Args:
            idx: (int) Index of dataset item to return.
        """
        return self.samples[idx]


def create_dataframes(file_paths_path: str, tracks_csv_path: str, genre_csv_path: str) -> \
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create dataframes for audio analysis.

    Args:
        file_paths_path: (pd.DataFrame) Path to 'all_data_paths.txt' storing the data paths for each sound file.
        tracks_csv_path: (pd.DataFrame) Path to 'tracks.csv' containing general track data.
        genre_csv_path: (str) Path to 'genre.csv' containing genre information.

    Returns:
        file_path_df: (pd.DataFrame) Dataframe storing the data paths for each sound file.
        track_df: (pd.DataFrame) Dataframe storing general track data.
        relevant_genre_df: (pd.DataFrame) Dataframe storing genre information.
        genre_df: (pd.DataFrame)
    """

    file_path_df = pd.read_csv(file_paths_path, header=None, names=['file_path'])

    track_df = pd.read_csv(tracks_csv_path, skiprows=[0, 1, 2],
                           usecols=[0, 6, 8, 11, 26, 39, 40, 41, 44, 47, 52],
                           names=['track_id', 'album_id', 'album_listens', 'album_title', 'artist_name',
                                  'track_favorites', 'track_genre_top', 'track_genres', 'track_interest',
                                  'track_listens', 'track_title'])

    track_df = track_df[track_df['track_genre_top'].notna()]

    def filter(row):
        return [int(i) for i in ast.literal_eval(row['track_genres'])]

    track_df['track_genres'] = track_df.apply(filter, axis=1)

    genre_df = pd.read_csv(genre_csv_path, usecols=[0, 3])
    relevant_indices = []
    for genre in track_df['track_genre_top'].unique():
        relevant_indices.append(genre_df[genre_df['title'] == genre].index.item())
    relevant_genre_df = genre_df.iloc[relevant_indices]
    relevant_genre_df.reset_index(inplace=True)

    return file_path_df, track_df, relevant_genre_df, genre_df


def create_dataset(file_path_df: pd.DataFrame, track_df: pd.DataFrame, genre_df: pd.DataFrame,
                   test_percentage: float = .10,
                   validation_percentage: float = .10) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create the custom dataset given the locations of the data.

    Args:
        file_path_df: (pd.DataFrame) Dataframe storing the data paths for each sound file.
        track_df: (pd.DataFrame) Dataframe storing general track data.
        genre_df: (pd.Dataframe) Dataframe storing genre information.
        test_percentage: (float) Percentage of paths to designate as part of the test dataset.
        validation_percentage: (float) Percentage of paths to designate as part of the validation dataset.

    Returns:
        train_data: (Dataset) Dataset containing training music data as spectrograms and genre labels.
        validation_data: (Dataset) Dataset containing validation music data as spectrograms and genre labels.
        test_data: (Dataset) Dataset containing testing music data as spectrograms and genre labels.
    """

    train_paths, test_paths = train_test_split(file_path_df, test_size=test_percentage)
    train_paths, validation_paths = train_test_split(train_paths, test_size=validation_percentage)

    train_data = SpectrogramDataset(train_paths, track_df, genre_df)

    print('Training dataset created.')

    validation_data = SpectrogramDataset(validation_paths, track_df, genre_df)

    print('Validation dataset created.')

    test_data = SpectrogramDataset(test_paths, track_df, genre_df)

    print('Test dataset created.')

    return train_data, validation_data, test_data


def create_mfcc_dataset(file_path_df: pd.DataFrame, track_df: pd.DataFrame, genre_df: pd.DataFrame,
                        test_percentage: float = .10,
                        validation_percentage: float = .10) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create the custom dataset given the locations of the data.

    Args:
        file_path_df: (pd.DataFrame) Dataframe storing the data paths for each sound file.
        track_df: (pd.DataFrame) Dataframe storing general track data.
        genre_df: (pd.Dataframe) Dataframe storing genre information.
        test_percentage: (float) Percentage of paths to designate as part of the test dataset.
        validation_percentage: (float) Percentage of paths to designate as part of the validation dataset.

    Returns:
        train_data: (Dataset) Dataset containing training music data as spectrograms and genre labels.
        validation_data: (Dataset) Dataset containing validation music data as spectrograms and genre labels.
        test_data: (Dataset) Dataset containing testing music data as spectrograms and genre labels.
    """

    train_paths, test_paths = train_test_split(file_path_df, test_size=test_percentage)
    train_paths, validation_paths = train_test_split(train_paths, test_size=validation_percentage)

    train_data = MfccDataset(train_paths, track_df, genre_df)

    print('Training dataset created.')

    validation_data = MfccDataset(validation_paths, track_df, genre_df)

    print('Validation dataset created.')

    test_data = MfccDataset(test_paths, track_df, genre_df)

    print('Test dataset created.')

    return train_data, validation_data, test_data


def create_audio_feature_dataset(file_path_df: pd.DataFrame, track_df: pd.DataFrame, genre_df: pd.DataFrame,
                                 test_percentage: float = .10,
                                 validation_percentage: float = .10, limit_samples=False, max_samples=100) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create the custom dataset given the locations of the data.

    Args:
        file_path_df: (pd.DataFrame) Dataframe storing the data paths for each sound file.
        track_df: (pd.DataFrame) Dataframe storing general track data.
        genre_df: (pd.Dataframe) Dataframe storing genre information.
        test_percentage: (float) Percentage of paths to designate as part of the test dataset.
        validation_percentage: (float) Percentage of paths to designate as part of the validation dataset.

    Returns:
        train_data: (Dataset) Dataset containing training music data as spectrograms and genre labels.
        validation_data: (Dataset) Dataset containing validation music data as spectrograms and genre labels.
        test_data: (Dataset) Dataset containing testing music data as spectrograms and genre labels.
    """

    train_paths, test_paths = train_test_split(file_path_df, test_size=test_percentage)
    train_paths, validation_paths = train_test_split(train_paths, test_size=validation_percentage)

    print("Starting making train_data")
    train_data = AudioFeature(train_paths, track_df, genre_df, limit_samples=limit_samples, max_samples=max_samples)

    print('Training dataset created.')

    validation_data = AudioFeature(validation_paths, track_df, genre_df, limit_samples=limit_samples, max_samples=10)

    print('Validation dataset created.')

    test_data = AudioFeature(test_paths, track_df, genre_df, limit_samples=limit_samples, max_samples=10)

    print('Test dataset created.')

    return train_data, validation_data, test_data

def load_audio_feature_dataset(train_samples_file, val_sample_file, test_sample_file):
    """
    Loads the custom dataset given the locations of the data.

    Args:
        ###ADD###

    Returns:
        train_data: (Dataset) Dataset containing training music data as spectrograms and genre labels.
        validation_data: (Dataset) Dataset containing validation music data as spectrograms and genre labels.
        test_data: (Dataset) Dataset containing testing music data as spectrograms and genre labels.
    """

    print("loading datasets")
    train_data = AudioFeature(samples = np.load(train_samples_file, allow_pickle = True))
    validation_data = AudioFeature(samples = np.load(val_sample_file, allow_pickle = True))
    test_data = AudioFeature(samples = np.load(test_sample_file, allow_pickle = True))

    return train_data, validation_data, test_data
