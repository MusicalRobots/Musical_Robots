"""Implementation of interactive class to communicate with user."""

import ipywidgets as widgets
import warnings
import ast

import pandas as pd

from DatasetQueries import *
from SVMPrediction import svm_prediction
from IPython.display import display, HTML, clear_output

from typing import Tuple


class Interactive:
    """Methods that create user interactive interfaces such as file uploader"""
    
    def __init__(self):
        self.most_similar_genres = None
        self.most_popular_song_ids = None
        self.most_popular_song = None
        self.most_popular_artist = None
        self.most_popular_album = None
        self.uploaded_filename = None
        self.y = None
        self.sr = None

        pass

    def uploader(self) -> Tuple[widgets.FileUpload, widgets.Output]:
        """
        Create and display a file uploader.

        Returns:
            uploader (widgets.FileUpload):
            output (widgets.Output):
        """
        
        uploader = widgets.FileUpload(multiple=False, description='Upload',
                                      layout=widgets.Layout(width='250px', height='40px'))
        output = widgets.Output()
        display(uploader, output)
        
        return uploader, output

    def if_play_songs_first(self):
        """
        Create and display yes-or-no buttons for users to choose if they want to listen to the song first.

        Args:
            y (np.ndarray): Array containing the audio time series.
            sr (int): The sampling rate of y.
        """
        b1 = widgets.Button(description='Yes', layout=widgets.Layout(width='30%'))
        b2 = widgets.Button(description='No', layout=widgets.Layout(width='30%'))
        display(HTML('<h2>Before analysing the genre, do you want to play your audio file?<h2>'))
        output = widgets.Output()
        h1 = widgets.HBox([b1, b2])
        display(h1, output)

        def yes_play_audio(b: widgets.Button):
            """If the users chooses yes, play the song."""

            b3 = widgets.Button(description='End', layout=widgets.Layout(width='30%'))
            output = widgets.Output()

            output.clear_output(wait=True)
            clear_output(wait=True)

            display(HTML('<h2>Gotcha! Gonna play the audio for you! <h2>'.format(b.description)))
            display(ipd.Audio(data=self.y, rate=self.sr))
            display(b3)

            b3.on_click(end_play_continue)

        def no_dont_play(b: widgets.Button):
            """
            If the user chooses no, skip playing the song and start analysing.

            Returns:
                 self.call_out_genre_prediction: The call out genre prediction method
            """
            output = widgets.Output()
            output.clear_output(wait=True)
            clear_output(wait=True)
            display(HTML('<h2>Ok, analysing the genre for you <h2>'.format(b.description)))

            return self.call_out_genre_prediction(output)

        def end_play_continue(b: widgets.Button):
            """
            Create and display an end play and continue analyzing button.

            After the users chooses to play the song, create and display an end button for
            users to choose if they want to end playing the song and continue on analysing

            Returns:
                self.call_out_genre_prediction: The call out genre prediction method
            """
            output = widgets.Output()
            output.clear_output(wait=False)
            clear_output(wait=False)
            display(HTML('<h2>Starting analysing...<h2>'))
            return self.call_out_genre_prediction(output)

        b1.on_click(yes_play_audio)
        b2.on_click(no_dont_play)

    def call_out_genre_prediction(self, output: widgets.Output) -> None:
        """

        Args:
            output (widgets.Output):
        """

        output.clear_output(wait=False)
        b4 = widgets.Button(description='Yes', layout=widgets.Layout(width='30%'))
        b5 = widgets.Button(description='No', layout=widgets.Layout(width='30%'))
        
        file_path_df = pd.read_csv('./data/file_path_df', index_col=0)
        track_df = pd.read_csv('./data/track_df', index_col=0)
        genre_df = pd.read_csv('./data/genre_df', index_col=0)
        total_genre_df = pd.read_csv('./data/total_genre_df', index_col=0)

        def filter(row: pd.Series) -> List[int]:
            """
            Create filter that can be applied to each row of a pandas dataframe.

            Filter changes string representation of a List in column 'track_genres' to a List[int].

            Args:
                row (pd.Series): Row of a pandas dataframe.
            Return:
                (List[int]): Literal evaluation of string representation of List.
            """
            return [int(i) for i in ast.literal_eval(row['track_genres'])]

        track_df['track_genres'] = track_df.apply(filter, axis=1)

        genre = svm_prediction(self.uploaded_filename, genre_df)
        
        display(HTML('<h2>The genre is ...{}<h2>'.format(genre)))
        display(HTML('<h2> Do you want to know the most popular song in that genre and similar genres?<h2>'))
        output = widgets.Output()
        h1 = widgets.HBox([b4, b5])
        display(h1, output)

        def yes_similiar_songs(b: widgets.Button):
            """If the users chooses yes, display the most popular songs and the most similar genres."""
            output = widgets.Output()
            output.clear_output(wait=True)
            clear_output(wait=False)

            self.most_popular_song_ids, self.most_popular_song, self.most_popular_artist, self.most_popular_album = \
                return_most_popular_song(genre=genre, genre_df=total_genre_df, track_df=track_df)
            display(HTML('<h3>The most popular songs in the genre {} are {} by {} '
                         'from the albums {} <h3>'.format(genre, ', '.join(self.most_popular_song),
                                                          ', '.join(self.most_popular_artist),
                                                          ', '.join(self.most_popular_album))))
            self.most_similar_genres = return_similar_genres(genre=genre, genre_df=total_genre_df, track_df=track_df, k=10)
            display(HTML('<h3>The most similar genres to {} are : {} <h3>'.format(genre,
                                                                                  ', '.join(self.most_similar_genres))))

            display(HTML('<h3> Would you like to hear the most popular song {} <h3><br>'.format(self.most_popular_song)))

            b6 = widgets.Button(description='Yes', layout=widgets.Layout(width='30%'))
            b7 = widgets.Button(description='No', layout=widgets.Layout(width='30%'))
            h2 = widgets.HBox([b6, b7])
            display(h2, output)

            b6.on_click(yes_play_song)
            b7.on_click(no_dont_play_song)

        def yes_play_song(b: widgets.Button):
            b8 = widgets.Button(description='End', layout=widgets.Layout(width='30%'))
            output = widgets.Output()

            output.clear_output(wait=True)
            clear_output(wait=True)

            display(HTML('<h2>Gotcha! Gonna playing the most popular song for you! <h2>'.format(b.description)))

            song_id = str(self.most_popular_song_ids[0]).zfill(6)
            file_path = file_path_df[file_path_df['file_path'].str.contains(song_id)]

            if len(file_path) != 0:
                filename = 'data/fma_small/' + file_path['file_path'].item()

                audio = play_song_from_filename(filename=filename)

                display(audio)
            else:
                display(HTML('<h2>Sorry, the audio could not be found! <h2>'.format()))

            display(b8)
            b8.on_click(end_play_song)

        def no_dont_play_song(b: widgets.Button):
            output = widgets.Output()
            output.clear_output(wait=True)
            clear_output(wait=True)
            display(HTML('<h2>Ok, would you like to hear a random song from one of the'
                         ' most similar genres: {} <h2>'.format(', '.join(self.most_similar_genres))))

            output = widgets.Output()
            h3 = widgets.HBox([b4, b5])
            display(h3, output)

            b4.on_click(yes_play_random_song_from_genre)
            b5.on_click(no_dont_play_random_song_from_genre)

        def end_play_song(b: widgets.Button):
            output = widgets.Output()
            output.clear_output(wait=True)
            clear_output(wait=True)
            display(HTML('<h2>Ok, would you like to hear a random song from one of the'
                         ' most similar genres: {} <h2>'.format(', '.join(self.most_similar_genres))))

            output = widgets.Output()
            h1 = widgets.HBox([b4, b5])
            display(h1, output)

            b4.on_click(yes_play_random_song_from_genre)
            b5.on_click(no_dont_play_random_song_from_genre)

        def yes_play_random_song_from_genre(b: widgets.Button):
            output = widgets.Output()
            output.clear_output(wait=True)
            clear_output(wait=True)

            display(HTML('<h2>Which genre would you like to hear a '
                         'random song from: {} ?<h2>'.format(', '.join(self.most_similar_genres))))

            s1=widgets.Select(
                    options=self.most_similar_genres,
                    rows=np.size(self.most_similar_genres),
                    description='genres:',
                    disabled=False)
            def play_random_song_from_selected_genre(change):
                output.clear_output(wait=False)
                clear_output(wait=False)
                
                random_song = play_random_song_from_genre(genre=s1.value,genre_df=total_genre_df,
                                                           track_df=track_df, path_df=file_path_df)

                if random_song is not None:
                    display(HTML('<h2> Playing song {} by {} from {} .<h2>'.format(random_song[1],
                                                                                   random_song[2],
                                                                                   random_song[3])))
                    display(random_song[0])

                else:
                    display(HTML('<h2> Sorry, could not find a song from {}.<h2>'.format(input_genre)))

                b9 = widgets.Button(description='Play another random song?', layout=widgets.Layout(width='30%'))
                b10 = widgets.Button(description='No.', layout=widgets.Layout(width='30%'))

                h4 = widgets.HBox([b9, b10])
                display(h4, output)

                b9.on_click(yes_play_random_song_from_genre)
                b10.on_click(no_dont_play_random_song_from_genre)
            
            display(s1,output)
            s1.observe(play_random_song_from_selected_genre,names='value')

        def no_dont_play_random_song_from_genre(b: widgets.Button):
            output.clear_output(wait=True)
            clear_output(wait=False)

            display(HTML("<h2>Hope it's helpful ! Thank you !<h2>"))
            pass

        def no_similiar_songs(b: widgets.Button) -> None:
            """If the users choose no, interaction ends"""

            output.clear_output(wait=True)
            clear_output(wait=False)

            display(HTML("<h2>Hope it's helpful ! Thank you !<h2>"))
            pass

        b4.on_click(yes_similiar_songs)
        b5.on_click(no_similiar_songs)

    def save_file(self, uploader: widgets.FileUpload):
        """
        save a copy of the uploaded file and load it using Librosa.

        Args:
            uploader (widgets.FileUpload):
        """

        self.uploaded_filename = next(iter(uploader.value))
        content = uploader.value[self.uploaded_filename]['content']

        # save the uploaded file to the same directory , and load it in librosa
        with open(f'./{self.uploaded_filename}', 'wb') as fp:
            fp.write(content)
        warnings.simplefilter('ignore')
        with warnings.catch_warnings(record=False):
            try:
                self.y, self.sr = librosa.load(self.uploaded_filename)

            except (RuntimeError, TypeError, audioread.NoBackendError):
                display(HTML('<h2>Sorry, the file you uploaded is not compatible, '
                             'please check your music/audio file and upload again<h2>'))
