"""Create a custom dataset that turns mp3 files into spectrograms."""

from pandas import pd
from torch.utils.data import Dataset
import warnings
import librosa
from typing import Tuple
from sklearn.model_selection import train_test_split
import ast

warnings.filterwarnings("ignore")


class SpectrogramDataset(Dataset):
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

            y, sr = librosa.load(filename, sr=None, mono=True)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)

            song_id = row[1].rsplit('/')[1].rsplit('.')[0].lstrip('0')
            genre_numeric = music_df[music_df['track_id'] == int(song_id)]['track_genres'].item()
            genre = genre_df[genre_df['genre_id'] == genre_numeric[0]]['title'].item()

            self.samples.append((mel, genre))

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[librosa.feature.melspectrogram, str]:
        """
        Return a single spectrogram and its label from the dataset.

        Args:
            idx: (int) Index of dataset item to return.
        """
        return self.samples[idx]


def create_dataset(file_paths_path: str, tracks_csv_path: str, genre_csv_path: str, test_percentage: float = .10,
                   validation_percentage: float = .10) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create the custom dataset given the locations of the data.

    Args:
        file_paths_path: (str) Path to 'all_data_paths.txt' storing the data paths for each sound file.
        tracks_csv_path: (str) Path to 'tracks.csv' containing general track data.
        genre_csv_path: (str) Path to 'genre.csv' containing genre information.
        test_percentage: (float) Percentage of paths to designate as part of the test dataset.
        validation_percentage: (float) Percentage of paths to designate as part of the validation dataset.

    Returns:
        train_data: (Dataset) Dataset containing training music data as spectrograms and genre labels.
        validation_data: (Dataset) Dataset containing validation music data as spectrograms and genre labels.
        test_data: (Dataset) Dataset containing testing music data as spectrograms and genre labels.
    """

    file_paths = pd.read_csv(file_paths_path, header=None, names=['file_path'])
    train_paths, test_paths = train_test_split(file_paths, test_size=test_percentage)
    train_paths, validation_paths = train_test_split(file_paths, test_size=validation_percentage)

    music_data = pd.read_csv(tracks_csv_path, skiprows=[0, 1, 2],
                             usecols=[0, 6, 8, 11, 26, 39, 41, 44, 47, 52],
                             names=['track_id', 'album_id', 'album_listens', 'album_title', 'artist_name',
                                    'track_favorites', 'track_genres', 'track_interest', 'track_listens',
                                    'track_title'])

    music_data['track_genres'] = music_data['track_genres'].apply(ast.literal_eval)

    genre_df = pd.read_csv(genre_csv_path, usecols=[0, 3])

    train_data = SpectrogramDataset(train_paths, music_data, genre_df)
    validation_data = SpectrogramDataset(validation_paths, music_data, genre_df)
    test_data = SpectrogramDataset(test_paths, music_data, genre_df)

    return train_data, validation_data, test_data
