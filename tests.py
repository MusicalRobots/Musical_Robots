import unittest
import pandas as pd
import librosa
import librosa.display

import SpectrogramDataset

class Testknn(unittest.TestCase):

    def test_load(self):
        # check that the data can load, (getting an error in Windows)
        # read input file
        file_paths = pd.read_csv('data/all_data_paths.txt', header=None, names=['file_path'])

        # find and load some file
        index = 7999
        filename = 'data/fma_small/' + file_paths['file_path'][index]
        y, sr = librosa.load(filename, sr=None, mono=True)

    def test_make_SpectrogramDataset(self):
        """ Test making a dataframe with a very shorted version of the data
        """
        file_path_df, track_df, genre_df = SpectrogramDataset.create_dataframes(file_paths_path = 'data/all_data_path_short.txt' ,
                                                     tracks_csv_path = 'data/fma_metadata/tracks_short.csv',
                                                     genre_csv_path = 'data/fma_metadata/genres.csv')

    # SpectrogramDataset(Dataset):
    #     """Create custom dataset of spectrograms and genre labels."""
    #
    #     def __init__(self, path_df: pd.DataFrame, music_df: pd.DataFrame, genre_df: pd.DataFrame) -> None:
    """
    def test_queries():
        try: #both should fail bc k not in (1,11)
            return_similar_genres(genre: 'pop', genre_df: pd.DataFrame, track_df: pd.DataFrame, k: int = 0)

        try: should fail bc k not in (1,11)
            return_similar_genres(genre: 'pop', genre_df: pd.DataFrame, track_df: pd.DataFrame, k: int = 11)

        try: genre should not exist
            return_similar_genres(genre='screeching cats', genre_df: pd.DataFrame, track_df: pd.DataFrame, k: int = 10)
            return_most_popular_song(genre: 'screeching cats', genre_df, track_df)
            play_random_song_from_genre(genre= 'screeching cats', genre_df, track_df, path_df)

        try: should return none bc audio not in dataset
            play_song_from_title(title = '#1', track_df, path_df)

            should play
            play_song_from_title(title= 'This World', track_df, path_df)
            should fail bc title does not exist
            play_song_from_title(title= 'Blah', track_df, path_df)
        """
