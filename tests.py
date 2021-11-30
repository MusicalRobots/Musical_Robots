"""
tests.py
Tests for Musical Robots
"""
import unittest
import pandas as pd
import librosa
import librosa.display

from SpectrogramDataset import *


class Tests(unittest.TestCase):
    """
    Test class for Musical Robots
    """

    def test_load(self):
        """ Test checks if files can load. """
        # check that the data can load, (getting an error in Windows)
        # read input file
        file_paths = pd.read_csv('data/all_data_path_short.txt', header=None,
                                names=['file_path'])

        # find and load some file
        index = 0
        filename = 'data/fma_small/' + file_paths['file_path'][index]
        librosa.load(filename, sr=None, mono=True)

    def test_make_SpectrogramDataset(self):
        """ Test making a dataframe with a very shorted version of the data
        """

        file_path_df, track_df, genre_df = create_dataframes(
                file_paths_path='data/all_data_path_short.txt',
                tracks_csv_path='data/fma_metadata/tracks_short.csv',
                genre_csv_path='data/fma_metadata/genres.csv')

    def test_split(self):
        """ test that the splitting function is working right"""
        file_path_df, track_df, genre_df = create_dataframes(
                file_paths_path='data/all_data_path_short.txt',
                tracks_csv_path='data/fma_metadata/tracks_short.csv',
                genre_csv_path='data/fma_metadata/genres.csv')

        #check that splitting is working right
        train, validate, test = train_validate_test_split(file_path_df, 0.2, 0.2)
        assert len(train) == 3
        assert len(validate) == 1
        assert len(test) == 1

    def test_audio_feature(self):
        """ Test making an audio feature object """
        file_path_df, track_df, genre_df = create_dataframes(
                file_paths_path='data/all_data_path_short.txt',
                tracks_csv_path='data/fma_metadata/tracks_short.csv',
                genre_csv_path='data/fma_metadata/genres.csv')
        #make 1 Audio data class
        train_data = AudioFeature(file_path_df, track_df, genre_df)





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
