"""Implementation of interactive class to communicate with user."""

import streamlit as st
import warnings
import ast
import audioread
import librosa
import pandas as pd

from dataset_queries import (
    return_similar_genres,
    return_most_popular_song,
    play_random_song_from_genre,
)
from svm_prediction import svm_prediction
from session_state import _get_state


class Interactive:

    """Methods that create user interactive interfaces."""

    def __init__(self):
        """
        Class to walk through interactions with the user.

        The user uploads an mp3 file.  They are given the option to listen
        to the audio.  Next, the genre is analyzed and the user is given
        the option to return the most popular song in the genre as well
        as the most similar genres.  The user is then given the options to
        listen to the most popular song as well as random songs from any
        of the most similar genres.  The interaction ends when the user no
        longer wants to listen to similar audio clips.

        """

        pass

    def start_up(self):
        """
        Start up the interaction.

        """
        st.title("Hi there. I'm the musical robot. Nice to meet you!")
        return self.upload_and_save()

    def upload_and_save(self):
        """
        Create and display a file uploader.

        """
        st.write(
            "If you want to know the genre of the music, please go ahead and "
            "upload your music/audio file below."
        )
        uploader = st.file_uploader(
            label="", key=st.session_state.upload_try_time,
            accept_multiple_files=False
        )

        if uploader is not None or st_state.uploaded:
            st_state.uploaded = True
            st.session_state.uploaded_filename = uploader.name
            st.session_state_content = uploader.getvalue()
            with open("./{}".format(st.session_state.uploaded_filename),
                      "wb") as fp:
                fp.write(st.session_state_content)
            warnings.simplefilter("ignore")
            with warnings.catch_warnings(record=False):
                try:
                    (y_audio, sample_rate) = librosa.load(
                        st.session_state.uploaded_filename
                    )
                    return self.if_play_songs_first()
                except (RuntimeError, TypeError, audioread.NoBackendError):
                    st.session_state.upload_try_time += 1
                    st.write(
                        "Sorry, the file you uploaded is not compatible, "
                        "please check your music/audio file and upload again"
                    )

                    return self.upload_and_save()
        else:
            st.stop()

    def if_play_songs_first(self):
        """
        Create and display yes-or-no buttons.
        Users choose if they want to listen to the song first.

        """

        st.title(
            "Before analysing the genre, "
            "do you want to play your audio file?"
        )

        if st.button("Yes", key=1) or st_state.yes_play_audio_first:
            st_state.yes_play_audio_first = True
            st.write("Gotcha! Gonna play the audio for you! ")
            st.audio(st.session_state_content)
            if st.button("Proceed", key=3) or st_state.proceed_1:
                st_state.proceed_1 = True
                st.write("Start analysing...")
                return self.call_out_genre_prediction()
            else:
                st.stop()
        elif st.button("No", key=2) or st_state.no_and_proceed:
            st_state.no_and_proceed = True
            st.write("Ok, analysing the genre for you")

            return self.call_out_genre_prediction()
        else:

            st.stop()

    def call_out_genre_prediction(self):
        """
        Return predicted song genre, similar genres, and most popular song.
        Args:

        """

        file_path_df = pd.read_csv("./data/file_path_df", index_col=0)
        track_df = pd.read_csv("./data/track_df", index_col=0)
        genre_df = pd.read_csv("./data/genre_df", index_col=0)
        total_genre_df = pd.read_csv("./data/total_genre_df", index_col=0)

        def row_filter(row: pd.Series):

            """
            Create filter that can be applied to each row of dataframe.
            Filter changes string representation of a List in column
            'track_genres' to a List[int].
            Args:
                row (pd.Series): Row of a pandas dataframe.
            Return:
                (List[int]): Literal evaluation of string representation
                of List.
            """

            return [int(i) for i in ast.literal_eval(row["track_genres"])]

        track_df["track_genres"] = track_df.apply(row_filter, axis=1)

        st.session_state.genre = svm_prediction(
            st.session_state.uploaded_filename, genre_df
        )
        st.title("***The genre is  {} !***".format(st.session_state.genre))
        st.write(
            "Do you want to know the most popular song in "
            "{} and similar genres?".format(
                st.session_state.genre
            )
        )
        st.session_state.file_path_df = file_path_df
        st.session_state.track_df = track_df
        st.session_state.genre_df = genre_df
        st.session_state.total_genre_df = total_genre_df

        if st.button("Yes", key=4) or st_state.yes_similar_songs:
            st_state.yes_similar_songs = True
            return self.yes_similar_songs(st.session_state.file_path_df)
        elif st.button("No", key=5) or st_state.no_and_end_interaction_1:
            st_state.no_and_end_interaction_1 = True
            return self.no_similar_songs()
        else:
            st.stop()

    def no_similar_songs(self):
        st.success("Hope this was helpful ! Thank you !")
        st.stop()
        pass

    def yes_similar_songs(self, file_path_df):
        """Display the most popular songs and the most similar genres."""

        (
            most_popular_song_ids,
            most_popular_song,
            most_popular_artist,
            most_popular_album,
        ) = return_most_popular_song(
            genre=st.session_state.genre,
            genre_df=st.session_state.total_genre_df,
            track_df=st.session_state.track_df,
        )
        st.write(
            "The most popular songs in the genre {} are "
            "{} by {} from the albums {}".format(
                st.session_state.genre,
                ", ".join(most_popular_song),
                ", ".join(most_popular_artist),
                ", ".join(most_popular_album),
            )
        )

        st.session_state.most_similar_genres = return_similar_genres(
            genre=st.session_state.genre,
            genre_df=st.session_state.total_genre_df,
            track_df=st.session_state.track_df,
            k=10,
        )
        st.write(
            "The most similar genres to {} are "
            ": {}".format(st.session_state.genre, ", ".join(
                    st.session_state.most_similar_genres)
            )
        )

        st.title(
            "Would you like to hear the most popular song {}".format(
                most_popular_song[0]
            )
        )

        if st.button("Yes", key=6) or st_state.yes_hear_most_popular_song:
            st_state.yes_hear_most_popular_song = True
            st.write("Gotcha! Gonna playing the most popular song for you! ")

            song_id = str(most_popular_song_ids[0]).zfill(6)
            file_path = file_path_df[file_path_df[
                "file_path"].str.contains(song_id)]

            if len(file_path) != 0:
                filename = "data/fma_small/" + file_path["file_path"].item()
                st.audio(filename)
                if st.button("Proceed", key=8) or st_state.proceed_2:
                    st_state.proceed_2 = True
                    return self.ask_if_hear_a_random_song()
                else:
                    st.stop()
            else:
                st.write("Sorry, the audio could not be found!".format())
        elif st.button("No", key=7) or st_state.no_ask_if_hear_random_song:
            st_state.no_ask_if_hear_random_song = True
            return self.ask_if_hear_a_random_song()
        else:
            st.stop()

    def ask_if_hear_a_random_song(self):
        st.write(
            "Ok, would you like to hear a random song from "
            "one of the most similar genres: {} ".format(
                ", ".join(st.session_state.most_similar_genres)
            )
        )

        if st.button("Yes") or st_state.yes_play_random_song_from_genre:
            st_state.yes_play_random_song_from_genre = True
            return self.yes_play_random_song_from_genre()
        elif st.button("No") or st_state.no_and_end_interaction_2:
            st_state.no_and_end_interaction_2 = True
            return self.end_interaction()
        else:
            st.stop()

    def end_interaction(self):
        st.success("Hope this was helpful ! Thank you !")
        st.stop()
        pass

    def yes_play_random_song_from_genre(self):
        st.write(
            "Which genre would you like to hear a random "
            "song from: {} ?".format(
                ", ".join(st.session_state.most_similar_genres)
            )
        )

        genre = st.selectbox(
            "genres", ["<select>"] + st.session_state.most_similar_genres, 0
        )
        if genre == "<select>":
            st.stop()
        else:
            random_song = play_random_song_from_genre(
                genre=genre,
                genre_df=st.session_state.total_genre_df,
                track_df=st.session_state.track_df,
                path_df=st.session_state.file_path_df,
            )
            if random_song is not None:
                st.write(
                    "Playing song {} by {} from {} .".format(
                        random_song[1], random_song[2], random_song[3]
                    )
                )

                st.write(random_song[0])
            else:

                st.write("Sorry, could not find a song from this genre.")

            if st.button("End interaction",
                         key=10) or st_state.end_interaction_3:
                st_state.end_interaction_3 = True
                return self.end_interaction()


if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.upload_try_time = 1
    st.session_state.uploaded_filename = None
    st.session_state_content = None
    st.session_state.genre = None
    st.session_state.track_df = None
    st.session_state.total_genre_df = None
    st.session_state.most_similar_genres = None
    st.session_state.file_path_df = None

robot = Interactive()

if st.session_state.initialized:
    st_state = _get_state()
    robot.start_up()
