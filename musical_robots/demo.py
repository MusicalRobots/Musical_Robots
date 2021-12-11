"""Implementation of interactive class to communicate with user."""
import streamlit as st
import ipywidgets as widgets
import warnings
import ast
import audioread
import librosa
import IPython.display as ipd
import pandas as pd
import numpy as np

from dataset_queries import return_similar_genres, return_most_popular_song, \
    play_random_song_from_genre, play_song_from_filename
from svm_prediction import svm_prediction
from IPython.display import display, HTML, clear_output

from typing import Tuple, List


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

        Attributes:
            self.most_similar_genres (List): List of most similar genres
            to the predicted genre.
            self.most_popular_song_ids (List): List of the unique ids of
            the most popular songs in the genre.
            self.most_popular_song (List): Title of the most popular song
            in the genre.
            self.most_popular_artist (List): Artist of the most popular song
            in the genre.
            self.most_popular_album (list): Album of the most popular song
            in the genre.
            self.uploaded_filename (str): Filename of the uploaded song.
            self.y_audio (np.ndarray) Audio time series of uploaded song.
            self.sample_rate (int) Sample rate of uploaded song.
        """
        #self.most_similar_genres = None
        #self.most_popular_song_ids = None
        #self.most_popular_song = None
        #self.most_popular_artist = None
        #self.most_popular_album = None
        #self.uploaded_filename = None
        #self.y_audio = None
        #self.sample_rate = None
        #self.music_file_content= None
        
        pass
    @st.cache(suppress_st_warning=True)
    def start_up(self):
        """
        Start up the interaction. Create and display a file uploader.
        
            
        """
        st.title("Hi there. I'm the musical robot. Nice to meet you!")
        st.write("If you want to know the genre of the music, please go ahead and upload your music/audio file below.")    
        
        uploader=st.file_uploader(label="",accept_multiple_files=False)
        
        if uploader is not None :
            st.session_state.uploaded_filename = uploader.name
            st.session_state_content = uploader.getvalue()
            with open(f"./{st.session_state.uploaded_filename}", "wb") as fp:
                fp.write(st.session_state_content)
            warnings.simplefilter("ignore")
            with warnings.catch_warnings(record=False):
                try:
                    y_audio, sample_rate = \
                        librosa.load(st.session_state.uploaded_filename)
                except (RuntimeError, TypeError, audioread.NoBackendError):
                    st.write(
                            "Sorry, the file you uploaded is not "
                            "compatible, please check your music/audio"
                            " file and upload again"
                        )
                    return self.upload_and_save()
        else:
            st.stop()
    @st.cache(suppress_st_warning=True)
    def if_play_songs_first(self):
        """
        Create and display yes-or-no buttons.
        Users choose if they want to listen to the song first.
        """
       
             
        st.title(
                "Before analysing the genre, do you want to play your"
                " audio file?"
            )
        if(st.button("Yes",key=1)):
            st.write(
                    "Gotcha! Gonna play the audio for you! "
                )
            st.audio(st.session_state_content)
            if(st.button("End",key=3)):
                st.write("Start analysing...")
            else:
                st.stop()
        elif(st.button("No",key=2)):
            st.write(
                    "Ok, analysing the genre "
                    "for you"
                )
        else:
            st.stop()
    @st.cache(suppress_st_warning=True)
    def call_out_genre_prediction(self) -> None:
        """
        Return predicted song genre, similar genres, and most popular song.
        Args:
            output (widgets.Output):
        """
        file_path_df = pd.read_csv("./data/file_path_df", index_col=0)
        track_df = pd.read_csv("./data/track_df", index_col=0)
        genre_df = pd.read_csv("./data/genre_df", index_col=0)
        total_genre_df = pd.read_csv("./data/total_genre_df", index_col=0)
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
        
        st.session_state.genre = svm_prediction(st.session_state.uploaded_filename, genre_df)
        st.title("***The genre is  {} !***".format(st.session_state.genre))
        st.write(
                "Do you want to know the most popular song in  "
                "{} and similar genres?".format (st.session_state.genre)
            )
        st.session_state.file_path_df = file_path_df
        st.session_state.track_df = track_df
        st.session_state.genre_df =genre_df
        st.session_state.total_genre_df=total_genre_df
        
        if (st.button("Yes",key=4)):
            pass
        elif (st.button("No",key=5)):
            return self.no_similar_songs()
        else:
            st.stop()
    @st.cache(suppress_st_warning=True)
    def no_similar_songs(self):
        st.success("Hope this was helpful ! Thank you !")
        st.stop()
        pass
    @st.cache(suppress_st_warning=True)
    def yes_similar_songs(self,file_path_df):
        """Display the most popular songs and the most similar genres."""
        most_popular_song_ids,most_popular_song,most_popular_artist, most_popular_album = return_most_popular_song(genre=st.session_state.genre, genre_df=st.session_state.total_genre_df, track_df=st.session_state.track_df)
        st.write(
                "The most popular songs in the genre {} are {} by {}"
                "from the albums {}".format(st.session_state.genre,  
                     ", ".join(most_popular_song),
                     ", ".join(most_popular_artist),
                     ", ".join(most_popular_album),
                 )
             )
        st.session_state.most_similar_genres = return_similar_genres(
             genre=st.session_state.genre, genre_df=st.session_state.total_genre_df, track_df=st.session_state.track_df, k=10
             )
        st.write("The most similar genres to {} are : {} <h3>".format(
               st.session_state.genre, ", ".join(st.session_state.most_similar_genres)
                 )
             )
        st.write("Would you like to hear the most popular song "
               "{}".format(most_popular_song[0])
             )

        if (st.button("Yes",key=6)):
            st.write("Gotcha! Gonna playing the most popular "
                 "song for you! "
             )
            song_id = str(most_popular_song_ids[0]).zfill(6)
            #file_path_df = st.session_state.file_path_df
            file_path = file_path_df[file_path_df["file_path"].str.contains(song_id)]

            if len(file_path) != 0:
                filename = '/mnt/e/fma_small/' + file_path['file_path'].item()
                st.audio(filename)
                if( st.button("End",key=8)):
                    pass
                else:
                    st.stop()
            else:
                st.write(
                        "Sorry, the audio could not be "
                        "found! <h2>".format()
                    )
        elif (st.button("No",key=7)):
            pass
        else:
            st.stop()
        
    def ask_if_hear_a_random_song(self):
        st.write("Ok, would you like to hear a random song from"
               " one of the most similar genres: {} <h2>".format(
               ", ".join(st.session_state.most_similar_genres)
                 )
             )
        if (st.button("Yes",key=9)):
            pass
        elif (st.button("No",key=10)):
            return self.end_interaction()
        else:
            st.stop()
    def end_interaction(self):
        st.success("Hope this was helpful ! Thank you !")
        st.stop()
        pass
        
    def yes_play_random_song_from_genre(self):
        st.write("Which genre would you like to hear a "
                  "random song from: {} ?<h2>".format(
                  ", ".join(st.session_state.most_similar_genres)
                )
            )
        
        genre = st.selectbox("genres",['<select>']+st.session_state.most_similar_genres,0)
        if genre == '<select>':
            st.stop()
        else:
            random_song = play_random_song_from_genre(
                 genre=genre,
                 genre_df=st.session_state.total_genre_df,
                 track_df=st.session_state.track_df,
                 path_df=st.session_state.file_path_df
                )
            if random_song is not None:
                st.write("Playing song {} by {} from {} .<h2>".format(
                             random_song[1], random_song[2], random_song[3]
                        )
                    )
                st.write(random_song[0])
            else:
                st.write(
                       "Sorry, could not find a song from this "
                       "genre."
                    )
            if (st.button("Play another random song?",key=11)):
                return self.yes_play_random_song_from_genre()
            if (st.button("No",key=12)):
                return self.no_dont_play_random_song_from_genre()
        

        
        
if not "initialized" in st.session_state:
    st.session_state.initialized = True
    st.session_state.uploaded_filename = None
    st.session_state_content = None
    st.session_state.genre = None
    st.session_state.track_df = None
    st.session_state.total_genre_df = None
    st.session_state.most_similar_genres = None
    st.session_state.file_path_df = None
    
robot=Interactive()
#col1, col2 = st.sidebar.columns([1, 1])

robot.start_up()

robot.if_play_songs_first()
robot.call_out_genre_prediction()
robot.yes_similar_songs(st.session_state.file_path_df)
robot.ask_if_hear_a_random_song()
robot.yes_play_random_song_from_genre()
robot.end_interaction()


#    def style_button_row(self, clicked_button_ix, n_buttons):
#        def get_button_indices(button_ix):
#            return {
#                'nth_child': button_ix,
#                'nth_last_child': n_buttons - button_ix + 1
#            }
#        clicked_style = """
#                    div[data-testid*="stHorizontalBlock"] > div:nth-child(%(nth_child)s): 
#                    nth-last-child(%(nth_last_child)s) button {
#                    border-color: "
#                    rgb(255, 75, 75);
#                    color: rgb(255, 75, 75);
#                    box-shadow: rgba(255, 75, 75, 0.5) 0px 0px 0px 0.2rem;
#                    outline: currentcolor none medium;
#                }
#                """
#        unclicked_style = """
#                    div[data-testid*="stHorizontalBlock"] > div:nth-child(%(nth_child)s):nth-last-child(%(nth_last_child)s) button {
#                    pointer-events: none;
#                    cursor: not-allowed;
#                    opacity: 0.65;
#                    filter: alpha(opacity=65);
#                    -webkit-box-shadow: none;
#                    box-shadow: none;
#                }
#                """
#        style = ""
#        for ix in range(n_buttons):
#            ix += 1
#            if ix == clicked_button_ix:
#                style += clicked_style % get_button_indices(ix)
#            else:
#                style += unclicked_style % get_button_indices(ix)
#        st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)