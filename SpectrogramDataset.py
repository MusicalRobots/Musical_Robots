"""Create a custom dataset that turns mp3 files into spectrograms."""

from pandas import pd
from torch.utils.data import Dataset
import warnings
import librosa

warnings.filterwarnings("ignore")


class SpectrogramDataset(Dataset):
    """Create custom dataset of spectrograms and genre labels."""

    def __init__(self, path_df: pd.DataFrame, music_df: pd.DataFrame, genre_df: pd.DataFrame):
        """
        Create list of data tuples.

        Args:
            path_df: (pd.Dataframe) DataFrame containing file paths.
            music_df: (pd.DataFrame) Dataframe containing music information.
            genre_df: (pd.DataFrame) Dataframe containing genre information.
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
