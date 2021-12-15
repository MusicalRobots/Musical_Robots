"""Implementation of interactive class to communicate with user."""

import streamlit as st
import warnings
import ast
import audioread
import librosa
import pandas as pd
import os

from dataset_queries import (
    return_similar_genres,
    return_most_popular_song,
    play_random_song_from_genre,
)

from svm_prediction import svm_prediction
from session_state import _get_state
from typing import List

data_path = os.path.join(os.getcwd(), "musical_robots/data/")


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
        st.markdown(
            """
            <h1>
            Hi there. I'm the musical robot 🤖
            Nice to meet you!
            </h1>
            """,
            unsafe_allow_html=True,
        )
        return self.upload_and_save()

    def upload_and_save(self):
        """
        Create and display a file uploader.

        """
        st.write(
            """
            If you want to know the genre of the music, please go ahead and
            upload your music/audio file below
            """
        )
        col1, col2 = st.columns(2)
        uploader = col1.file_uploader(
            label="", key=st.session_state.upload_try_time,
            accept_multiple_files=False
        )
        if uploader is not None or st_state.uploaded:
            st_state.uploaded = True
            if st_state.uploader_proceed:
                return self.if_play_songs_first()
                st.stop()
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
                    st_state.uploader_proceed = True
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

        st.markdown(
            """
            <h4>
            Before analysing the genre,
            do you want to play your audio file?
            </h4>
            """,
            unsafe_allow_html=True,
        )
        col2 = st.columns(5)
        if col2[0].button("Yes", key=1) or st_state.yes_play_audio_first:
            st_state.yes_play_audio_first = True
            st.audio(st.session_state_content)
            if st.button("Proceed", key=3) or st_state.proceed_1:
                st_state.proceed_1 = True
                return self.call_out_genre_prediction()
            else:
                st.stop()
        elif col2[1].button("No", key=2) or st_state.no_and_proceed_1:
            st_state.no_and_proceed_2 = True
            return self.call_out_genre_prediction()
        else:

            st.stop()

    def call_out_genre_prediction(self):
        """
        Return predicted song genre, similar genres, and most popular song.
        Args:

        """

        file_path_df = pd.read_csv(os.path.join(data_path, 'file_path_df'),
                                   index_col=0)
        track_df = pd.read_csv(os.path.join(data_path, "track_df"),
                               index_col=0)
        genre_df = pd.read_csv(os.path.join(data_path, "genre_df"),
                               index_col=0)
        total_genre_df = pd.read_csv(os.path.join(data_path, "total_genre_df"),
                                     index_col=0)

        def row_filter(row: pd.Series) -> List[int]:

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
            st.session_state.uploaded_filename, genre_df,
            os.path.join(data_path, "svm_model.pkl")
        )
        st.title("***The genre is  {} !***".format(st.session_state.genre))
        st.markdown(
            """
            <h4>
            Do you want to know the most popular song in
            {} and similar genres?
            </h4>
             """.format(
                st.session_state.genre
            ),
            unsafe_allow_html=True,
        )
        st.session_state.file_path_df = file_path_df
        st.session_state.track_df = track_df
        st.session_state.genre_df = genre_df
        st.session_state.total_genre_df = total_genre_df
        col3 = st.columns(5)
        if col3[0].button("Yes", key=4) or st_state.yes_similar_songs:
            st_state.yes_similar_songs = True
            return self.yes_similar_songs(st.session_state.file_path_df)
        elif col3[1].button("No", key=5) or st_state.no_and_end_interaction_1:
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
        st.markdown(
            """
            <h4>
            The most popular songs in the genre
            {} are {} by {} from the albums {}.
            </h4>
            """.format(
                st.session_state.genre,
                ", ".join(most_popular_song),
                ", ".join(most_popular_artist),
                ", ".join(most_popular_album),
            ),
            unsafe_allow_html=True,
        )

        st.session_state.most_similar_genres = return_similar_genres(
            genre=st.session_state.genre,
            genre_df=st.session_state.total_genre_df,
            track_df=st.session_state.track_df,
            k=10,
        )

        def ulify(elements: List[str]) -> str:
            """
            Create and return a string of unordered list
            that works in HTML
            Args:
                elements: An numpy list
            Return:
                (str): a string with HTML unordered list
                components.
            """
            string = "<ul>\n"
            for s in elements:
                string += "<li>" + str(s) + "</li>\n"
            string += "</ul>"
            return string

        st.markdown(
            """
            <h4>The most similar genres to {} are:<br>{}</h4>
             """.format(
                st.session_state.genre, ulify(
                    st.session_state.most_similar_genres)
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <h4>Would you like to hear the most popular song {} ?</h4>
            """.format(
                most_popular_song[0]
            ),
            unsafe_allow_html=True,
        )
        col4 = st.columns(5)
        if col4[0].button("Yes", key=6) or st_state.yes_hear_most_popular_song:
            st_state.yes_hear_most_popular_song = True
            song_id = str(most_popular_song_ids[0]).zfill(6)
            file_path = file_path_df[file_path_df[
                "file_path"].str.contains(song_id)]

            if len(file_path) != 0:
                # filename = "data/fma_small/" + file_path["file_path"].item()
                filename = os.path.join(data_path, "fma_small/"
                                        + file_path["file_path"].item())
                st.audio(filename)
                if st.button("Proceed", key=8) or st_state.proceed_2:
                    st_state.proceed_2 = True
                    return self.ask_if_hear_a_random_song()
                else:
                    st.stop()
            else:
                st.write("Sorry, the audio could not be found!".format())
        elif col4[1].button("No", key=7) or st_state.no_and_proceed_2:
            st_state.no_and_proceed_2 = True
            return self.ask_if_hear_a_random_song()
        else:
            st.stop()

    def ask_if_hear_a_random_song(self):
        st.markdown(
            """
            <h4>
            Ok, would you like to hear a random song from
            one of the most similar genres?
            </h4>
            """,
            unsafe_allow_html=True,
        )

        col5 = st.columns(5)
        if col5[0].button("Yes") or st_state.yes_play_random_song_from_genre:
            st_state.yes_play_random_song_from_genre = True
            return self.yes_play_random_song_from_genre()
        elif col5[1].button("No") or st_state.no_and_end_interaction_2:
            st_state.no_and_end_interaction_2 = True
            return self.end_interaction()
        else:
            st.stop()

    def end_interaction(self):
        st.balloons()
        st.success("Hope this was helpful ! Thank you !")
        st.stop()
        pass

    def yes_play_random_song_from_genre(self):
        st.markdown(
            """
            <h4>
            Which genre would you like to hear a random
            song from ?
            </h4>
            """,
            unsafe_allow_html=True,
        )
        col6 = st.columns(2)
        genre = col6[0].selectbox(
            "", ["<select>"] + st.session_state.most_similar_genres, 0
        )
        if genre == "<select>":
            st.stop()
        else:
            random_song = play_random_song_from_genre(
                genre=genre,
                genre_df=st.session_state.total_genre_df,
                track_df=st.session_state.track_df,
                path_df=st.session_state.file_path_df,
                path_to_data=os.path.join(data_path, "fma_small/")
            )
            if random_song[0] is not None:
                st.write(
                    "Playing song {} by {} from {} .".format(
                        random_song[1], random_song[2], random_song[3]
                    )
                )
                play_song = st.empty()
                play_song.write(random_song[0])
            else:

                st.write("Sorry, could not find a song from this genre.")
            if st.button("End interaction",
                         key=10) or st_state.end_interaction_3:
                st_state.end_interaction_3 = True
                return self.end_interaction()


if __name__ == '__main__':
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
        st.set_page_config(
            page_title="Musical Robot Cool Web",
            page_icon="🤖", layout="centered",
        )
        robot.start_up()
