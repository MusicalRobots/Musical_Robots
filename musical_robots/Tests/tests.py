"""
tests.py
Tests for Musical Robots
"""
import unittest
import pandas as pd

from musical_robots.spectrogram_dataset import create_dataframes, AudioFeature, split_data, \
                               create_audio_feature_dataset
from musical_robots.svm_prediction import svm_prediction
from musical_robots.dataset_queries import return_similar_genres, return_most_popular_song, \
    play_random_song_from_genre, play_song_from_filename

import warnings

class Tests(unittest.TestCase):
    """
    Test class for Musical Robots
    """
    @classmethod
    def setUpClass(cls):
        """
            Test that we can create a dataframe and then save results for
            other tests
        """

        warnings.filterwarnings("ignore")

        output = create_dataframes(file_paths_path='musical_robots/data/all_data_path_short.txt',
                                   tracks_csv_path='musical_robots/data/fma_metadata/tracks_short.csv',
                                   genre_csv_path='musical_robots/data/fma_metadata/genres.csv')

        cls.file_path_df, cls.track_df, cls.relevant_genre_df, cls.genre_df = output

        file = cls.file_path_df.iloc[0]['file_path']
        cls.filename = "musical_robots/data/fma_small/" + file

        cls.genre = svm_prediction(filename=cls.filename, genre_df=cls.relevant_genre_df,
                                    model_filename='musical_robots/svm_model.pkl')

    def test_create_dataframes(self):
        """Test creating dataframes with a very shorted version of the data."""
        output = create_dataframes(file_paths_path='musical_robots/data/all_data_path_short.txt',
                                   tracks_csv_path='musical_robots/data/fma_metadata/tracks_short.csv',
                                   genre_csv_path='musical_robots/data/fma_metadata/genres.csv')
        assert len(output) == 4

    def test_split(self):
        """ test that the splitting function is working right"""
        train, test, validate = split_data(self.file_path_df, 0.2, 0.2)
        assert len(train) == 3
        assert len(validate) == 1
        assert len(test) == 1

    def test_audio_feature_dataset(self):
        """Smoke test to test the creation of an audio feature dataset."""
        create_audio_feature_dataset(self.file_path_df,
                                     self.track_df,
                                     self.genre_df,
                                     path_to_data="musical_robots/data/fma_small/",
                                     test_percentage=0.2,
                                     validation_percentage=0.2)

    def test_audio_feature(self):
        """Test making an audio feature object """
        AudioFeature(path_to_data="musical_robots/data/fma_small/", path_df=self.file_path_df,
                     music_df=self.track_df, genre_df=self.genre_df)

    def test_audio_feature_fake_file(self):
        """Test audio feature with fake filepath."""
        fake_df = pd.DataFrame(data={'col1': ['fake_file']})
        empty_dataset = AudioFeature(path_to_data="musical_robots/data/fma_small/", path_df=fake_df,
                                     music_df=self.track_df, genre_df=self.genre_df)
        self.assertEqual(empty_dataset.__len__(), 0)

    def test_prediction(self):
        """Smoke test to see if svm prediction works."""
        file = self.file_path_df.iloc[0]['file_path']
        filename = "musical_robots/data/fma_small/" + file

        genre = svm_prediction(filename=self.filename, genre_df=self.relevant_genre_df,
                               model_filename='musical_robots/svm_model.pkl')

    def test_prediction_fake_file(self):
        """Test prediction with a fake file."""
        genre = svm_prediction(filename='fake_file', genre_df=self.relevant_genre_df,
                               model_filename='musical_robots/svm_model.pkl')
        self.assertIsNone(genre)

    # def test_prediction_audioread_error(self):
    #     #audioread

    def test_similar_genres(self):
        """
            Smoke tests to test functions of data queries
        """
        k = 10
        similar_genres = return_similar_genres(genre=self.genre, genre_df=self.genre_df, track_df=self.track_df, k=k)
        assert len(similar_genres) < k

    def test_similar_genres_not_a_genre(self):
        """
            Test to check that fake genre throws an exception.
        """
        with self.assertRaises(Exception):
            return_similar_genres(genre='foo', genre_df=self.genre_df, track_df=self.track_df, k=10)

    def test_similar_genres_k_out_of_range(self):
        """
            Test to check that k out of range throws an exception.
        """
        with self.assertRaises(Exception):
            #when k is too small
            return_similar_genres(genre='pop', genre_df=self.genre_df, track_df=self.track_df, k=0)
            #when k is too big
            return_similar_genres(genre='pop', genre_df=self.genre_df, track_df=self.track_df, k=11)

    def test_most_popular(self):
        """Smoke tests to test functions of data queries."""
        return_most_popular_song(genre=self.genre, genre_df=self.genre_df, track_df=self.track_df)

    def test_most_popular_fake_genre(self):
        """Test fake genre."""
        with self.assertRaises(Exception):
            return_most_popular_song(genre="foo", genre_df=self.genre_df, track_df=self.track_df)

    def test_play_random_song_from_genre(self):
        """Smoke test for play random song from genre."""
        play_random_song_from_genre(genre=self.genre, genre_df=self.genre_df, track_df=self.track_df,
                                    path_df=self.file_path_df, path_to_data='musical_robots/data/fma_small/')

    def test_play_random_song_from_genre_no_songs(self):
        """Test that function runs when the genre exists, but there is no corresponding audio."""
        output = play_random_song_from_genre(genre="nu-jazz", genre_df=self.genre_df, track_df=self.track_df,
                                             path_df=self.file_path_df, path_to_data='musical_robots/data/fma_small/')
        self.assertIsNone(output[0])

    def test_play_random_song_from_genre_fake_genre(self):
        """Test for when genre does not exist in dataset."""
        output =play_random_song_from_genre(genre="foo", genre_df=self.genre_df, track_df=self.track_df,
                                            path_df=self.file_path_df, path_to_data='musical_robots/data/fma_small/')
        self.assertIsNone(output[0])

    # def test_play_random_song_from_genre_audioread(self):
    #     #audioread

    def test_play_song_from_filename(self):
        """Test that audio is returned for a valid filename."""
        audio = play_song_from_filename(self.filename)
        self.assertIsNotNone(audio)

    def test_play_song_invalid_filename(self):
        """Test filename does not exist."""
        audio = play_song_from_filename('fake_filename')
        self.assertIsNone(audio)

    # def test_play_song_from_filename_audioread(self):
    #     #audioread
