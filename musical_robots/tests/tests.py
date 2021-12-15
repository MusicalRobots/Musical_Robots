# pylint: disable=R0904
"""
tests.py
Tests for Musical Robots
"""
import os
import warnings
import unittest
import pandas as pd
import numpy as np
import musical_robots

from musical_robots.spectrogram_dataset import (
    create_dataframes,
    AudioFeature,
    split_data,
    create_audio_feature_dataset,
)
from musical_robots.svm_prediction import svm_prediction, \
    svm_accuracy_report

from musical_robots.dataset_queries import (
    return_similar_genres,
    return_most_popular_song,
    play_random_song_from_genre,
    play_song_from_filename,
)

from musical_robots.demo import Interactive

data_path = os.path.join(musical_robots.__path__[0], "data/fma_small/")


class Tests(unittest.TestCase):
    """
    Test class for Musical Robots
    """

    @classmethod
    def setUpClass(cls):
        """Test that we can create a dataframe and then save results."""

        warnings.filterwarnings("ignore")

        output = create_dataframes(
            file_paths_path="musical_robots/data/all_data_path_short.txt",
            tracks_csv_path="musical_robots/data/fma_metadata/"
                            "tracks_short.csv",
            genre_csv_path="musical_robots/data/fma_metadata/genres.csv",
        )
        assert len(output) == 4

        cls.file_path_df, cls.track_df, cls.relevant_genre_df,\
            cls.genre_df = output

    # def test_create_dataframes(self):
    #     """Test creating dataframes with shorted version of the data."""
    #     output = create_dataframes(file_paths_path='musical_robots/data/
    #     all_data_path_short.txt',
    #                                tracks_csv_path='musical_robots/data/fma_metadata
    #                                /tracks_short.csv',
    #                                genre_csv_path='musical_robots/data/fma_metadata/genres.csv')
    #     assert len(output) == 4

    def test_split(self):
        """test that the splitting function is working right"""
        train, test, validate = split_data(self.file_path_df, 0.2, 0.2)
        assert len(train) == 3
        assert len(validate) == 1
        assert len(test) == 1

    def test_audio_feature_dataset(self):
        """Smoke test to test the creation of an audio feature dataset."""
        create_audio_feature_dataset(
            self.file_path_df,
            self.track_df,
            self.genre_df,
            path_to_data=data_path,
            test_percentage=0.2,
            validation_percentage=0.2,
        )

    def test_audio_feature(self):
        """Test making an audio feature object"""
        afd = AudioFeature(
            path_to_data=data_path,
            path_df=self.file_path_df,
            music_df=self.track_df,
            genre_df=self.genre_df,
        )

        assert afd.__len__() == 5
        self.assertIsNotNone(afd.__getitem__(0))

    def test_audio_feature_fake_file(self):
        """Test audio feature with fake filepath."""
        fake_df = pd.DataFrame(data={"col1": ["fake_file"]})
        empty_dataset = AudioFeature(
            path_to_data="musical_robots/data/fma_small/",
            path_df=fake_df,
            music_df=self.track_df,
            genre_df=self.genre_df,
        )
        self.assertEqual(empty_dataset.__len__(), 0)

    def test_prediction(self):
        """Smoke test to see if svm prediction works."""
        filename = os.path.join(data_path, "000/000002.mp3")
        genre = svm_prediction(
            filename=filename,
            genre_df=self.relevant_genre_df,
            model_filename="musical_robots/data/svm_model.pkl",
        )
        self.assertIsNotNone(genre)

    def test_prediction_fake_file(self):
        """Test prediction with a fake file."""
        genre = svm_prediction(
            filename="fake_file",
            genre_df=self.relevant_genre_df,
            model_filename="musical_robots/data/svm_model.pkl",
        )
        self.assertIsNone(genre)

    def test_prediction_audioread_error(self):
        """Test prediction of real file with audio read error."""
        filename = os.path.join(data_path, "108/108925.mp3")
        genre = svm_prediction(
            filename=filename,
            genre_df=self.relevant_genre_df,
            model_filename="musical_robots/data/svm_model.pkl",
        )
        self.assertIsNone(genre)

    def test_svm_accuracy_report_invalid_inputs(self):
        """Test the svm accuracy report with invalid inputs."""
        with self.assertRaises(Exception):
            svm_accuracy_report(true_labels=None, pred_labels=np.arange(5))
        with self.assertRaises(Exception):
            svm_accuracy_report(true_labels=np.arange(5), pred_labels=None)
        with self.assertRaises(Exception):
            svm_accuracy_report(true_labels=np.arange(5),
                                pred_labels=np.arange(3))

    def test_svm_accuracy_report(self):
        tp_rate, fp_rate, accuracy = \
            svm_accuracy_report(true_labels=[1, 2, 3, 2, 3, 2, 1],
                                pred_labels=[1, 2, 2, 2, 3, 2, 3])
        true_tp_rate = [.50, 1, .50]
        true_fp_rate = [0, .25, .20]

        np.testing.assert_array_equal(tp_rate, true_tp_rate)
        np.testing.assert_array_equal(fp_rate, true_fp_rate)
        self.assertAlmostEqual(71, accuracy, places=0)

    def test_similar_genres(self):
        """Smoke tests to test functions of data queries."""
        k = 10
        similar_genres = return_similar_genres(
            genre="pop", genre_df=self.genre_df, track_df=self.track_df, k=k
        )
        assert len(similar_genres) < k

    def test_similar_genres_not_a_genre(self):
        """Test to check that fake genre throws an exception."""
        with self.assertRaises(Exception):
            return_similar_genres(
                genre="foo", genre_df=self.genre_df,
                track_df=self.track_df, k=10
            )

    def test_similar_genres_k_out_of_range(self):
        """Test to check that k out of range throws an exception."""
        with self.assertRaises(Exception):
            return_similar_genres(
                genre="pop", genre_df=self.genre_df,
                track_df=self.track_df, k=0
            )
        with self.assertRaises(Exception):
            return_similar_genres(
                genre="pop", genre_df=self.genre_df,
                track_df=self.track_df, k=11
            )

    def test_similar_genres_invalid_args(self):
        """Test invalid arguments in return similar genres."""
        with self.assertRaises(Exception):
            return_similar_genres(
                genre=None, genre_df=self.genre_df,
                track_df=self.track_df, k=10
            )
        with self.assertRaises(Exception):
            return_similar_genres(
                genre="pop", genre_df=None, track_df=self.track_df, k=10
            )
        with self.assertRaises(Exception):
            return_similar_genres(
                genre="pop", genre_df=self.genre_df, track_df=None, k=10
            )

    def test_most_popular(self):
        """Smoke tests to test functions of data queries."""
        return_most_popular_song(
            genre="pop", genre_df=self.genre_df, track_df=self.track_df
        )

    def test_most_popular_fake_genre(self):
        """Test fake genre."""
        with self.assertRaises(RuntimeError):
            return_most_popular_song(
                genre="foo", genre_df=self.genre_df, track_df=self.track_df
            )

    def test_play_random_song_from_genre(self):
        """Smoke test for play random song from genre."""
        play_random_song_from_genre(
            genre="pop",
            genre_df=self.genre_df,
            track_df=self.track_df,
            path_df=self.file_path_df,
            path_to_data=data_path,
        )

    def test_play_random_song_from_genre_no_songs(self):
        """Test when the genre exists, but there is no corresponding audio."""
        song_output = play_random_song_from_genre(
            genre="nu-jazz",
            genre_df=self.genre_df,
            track_df=self.track_df,
            path_df=self.file_path_df,
            path_to_data=data_path,
        )
        self.assertIsNone(song_output[0])

    def test_play_random_song_from_genre_fake_genre(self):
        """Test for when genre does not exist in dataset."""
        song_output = play_random_song_from_genre(
            genre="foo",
            genre_df=self.genre_df,
            track_df=self.track_df,
            path_df=self.file_path_df,
            path_to_data=data_path,
        )
        self.assertIsNone(song_output[0])

    def test_play_random_song_from_genre_audioread(self):
        """Test play random song with real file with audioread error."""
        track_df = pd.DataFrame(data={"track_genres": [[25]],
                                      "track_id": ["108925"]})
        path_df = pd.DataFrame(data={"file_path": ["108/108925.mp3"]})
        song_output = play_random_song_from_genre(
            genre="punk",
            genre_df=self.genre_df,
            track_df=track_df,
            path_df=path_df,
            path_to_data=data_path,
        )
        self.assertIsNone(song_output[0])

    def test_play_random_song_from_genre_invalid_args(self):
        """Test play random song with invalid arguments."""
        with self.assertRaises(Exception):
            play_random_song_from_genre(
                genre=None,
                genre_df=self.genre_df,
                track_df=self.track_df,
                path_df=self.file_path_df,
                path_to_data=data_path,
            )
        with self.assertRaises(Exception):
            play_random_song_from_genre(
                genre="pop",
                genre_df=None,
                track_df=self.track_df,
                path_df=self.file_path_df,
                path_to_data=data_path,
            )
        with self.assertRaises(Exception):
            play_random_song_from_genre(
                genre="pop",
                genre_df=self.genre_df,
                track_df=None,
                path_df=self.file_path_df,
                path_to_data=data_path,
            )
        with self.assertRaises(Exception):
            play_random_song_from_genre(
                genre="pop",
                genre_df=self.genre_df,
                track_df=self.track_df,
                path_df=None,
                path_to_data=data_path,
            )

    def test_play_song_from_filename(self):
        """Test that audio is returned for a valid filename."""
        filename = os.path.join(data_path, "000/000002.mp3")
        audio = play_song_from_filename(filename)
        self.assertIsNotNone(audio)

    def test_play_song_invalid_filename(self):
        """Test filename does not exist."""
        audio = play_song_from_filename("fake_filename")
        self.assertIsNone(audio)

    def test_play_song_from_filename_audioread(self):
        """Test play song with real file with audioread error."""
        filename = os.path.join(data_path, "108/108925.mp3")
        audio = play_song_from_filename(filename)
        self.assertIsNone(audio)

    def test_interactive(self):
        """Smoke test for the interactive class"""
        Interactive()
