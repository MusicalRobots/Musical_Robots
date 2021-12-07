"""
tests.py
Tests for Musical Robots
"""
import unittest

from musical_robots.spectrogram_dataset import create_dataframes, AudioFeature, split_data, \
                               create_audio_feature_dataset
from musical_robots.svm_prediction import svm_prediction
from musical_robots.dataset_queries import return_similar_genres, return_most_popular_song

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
        assert len(output) == 4
        cls.file_path_df, cls.track_df, cls.relevant_genre_df, cls.genre_df = output

    # def test_make_spectrogram_dataset(self):
    #     """ Test making a dataframe with a very shorted version of the data
    #     """
    #
    #     output = create_dataframes(file_paths_path='data/all_data_path_short.txt',
    #                                tracks_csv_path='data/fma_metadata/tracks_short.csv',
    #                                genre_csv_path='data/fma_metadata/genres.csv')
    #     assert len(output) == 4

    def test_all_loaded(self):
        """ Test that all the data files were loaded
        """

        assert len(self.file_path_df) == len(self.track_df)


    def test_split(self):
        """ test that the splitting function is working right"""

        #check that splitting is working right
        # train, validate, test = train_validate_test_split(file_path_df, 0.2, 0.2)
        train, test, validate = split_data(self.file_path_df, 0.2, 0.2)
        assert len(train) == 3
        assert len(validate) == 1
        assert len(test) == 1

    def test_audio_feature_dataset(self):
        """
            Smoke test to test the creation of an audio feature dataset
        """

        create_audio_feature_dataset("musical_robots/data/fma_small/",
                                     self.file_path_df,
                                     self.track_df,
                                     self.genre_df,
                                     test_percentage=0.2,
                                     validation_percentage=0.2)


    def test_audio_feature(self):
        """ Test making an audio feature object """

        #make 1 Audio data class
        AudioFeature("musical_robots/data/fma_small/", self.file_path_df, self.track_df, self.genre_df)

    def test_prediction(self):
        """ Smoke test to see if svm prediction works
        """

        file = self.file_path_df.iloc[0]['file_path']
        filename = "musical_robots/data/fma_small/" + file

        genre = svm_prediction(filename, self.genre_df,
                               model_filename='musical_robots/svm_model.pkl')



    def test_similar_genres(self):
        """
            Smoke tests to test functions of data queries
        """
        # filename = './data/fma_small/000/000002.mp3'

        # genre = svm_prediction(filename, self.genre_df)

        k = 10
        similar_genres = return_similar_genres('pop', self.genre_df, self.track_df, k)
        assert len(similar_genres) < k

    def test_similar_genres_not_a_genre(self):
        """
            Test to check that fake genre throws an exception
        """
        with self.assertRaises(Exception):
            return_similar_genres('foo', self.genre_df, self.track_df, k=10)

    def test_similar_genres_k_outofrange(self):
        """
            Test to check that fake genre throws an exception
        """
        with self.assertRaises(Exception):
            #when k is too small
            return_similar_genres('foo', self.genre_df, self.track_df, k=0)
            #when k is too big
            return_similar_genres('foo', self.genre_df, self.track_df, k=11)

    def test_most_pop(self):
        """
            Smoke tests to test functions of data queries
        """
        # filename = './data/fma_small/000/000002.mp3'# + file

        # genre = svm_prediction(filename, self.genre_df)

        return_most_popular_song('pop', self.genre_df, self.track_df)
        # assert len(most_popular_song) == 1







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
