# User Stories

## User 1

Ron has an audio file.  Ron wants to know the genre of the music file. Ron is in CSE 583 and knows what a Jupyter Notebook is.  Ron uploads his music file, and runs some commands in the notebook to return the genre.  Ron decided he wants to know the most popular songs in that genre.  He asks the system to return them.  Ron also wants to know what similar genres exist.  The system returns them.

**User:** Ron uploads a music file.  
**System:** Returns a figure displaying the probabilities that a clip belongs to genres.  The system also returns a message declaring the most likely genre. 
Display "Do you want to know the most popular songs in the genre?"
**Ron:** Says yes or no.
**System:** Display "Do you want to know the most similar genres?"
**Ron:** Says yes or no.
**System:** Display "Do you want to try another audio file"?
**Ron:** Says yes or no.


## User 2

Valentina wants to change the architecture of the network used to predict music genre and/or add data to the training set.  Valentina is a data scientist and highly technical.  

## User 3

User wants to use our service.  Has a bunch of files.  Doesn't want to deal with a NN. 

# Software Components

## Use case: return most similar genres
Name: ReturnSimilarGenres

What it does: Returns up to top 10 most similar genres to the input genre ranked by decreasing similarity.

Inputs: genre: _string_ specifying genre to compare to
	number_similar: _integer_ between 1 and 10 specifying how many ranked similar genres to return

Outputs: similar_genres: _List of strings_ containing ranked list of most similar genres 

## Use Case: Play song
Name: play_song

What it does: Plays an mp3 file.

Inputs: song_mp3: _mp3_ of a song 

Outputs: _IPython.core.display.DisplayObject_ The playable file displayed in the Jupyter notebook.

## Use Case: Return mp3 of a song.

Name: return_mp3

What it does: Queries the song database to see if a particular mp3 is in the database.  If the song exists, the mp3 is returned.  If the file does not exist, the user is told the song does not exist in the database.

Inputs: song_name: _string_ Title of a song

Outputs: song_mp3: _mp3_ of a the given song 

## Use Case: Return most popular song in a genre
Name: return_most_popular_song

What it does: Queries the database containing songs and their respective genres to return the most popular song in a given genre.

Inputs: genre: _string_ genre name

Outputs: most_popular_song: _string_ Name of most popular song in the genre.

## Use case: Return most similar genre
Name: return_most_similar_genre

What it does: Queries the genre and song databases to find the genres most often co-classifed with the given genre.

Inputs: genre: _string_ genre name

Outputs: most_similar_genres: _List of strings_ Ranked list of most similar genres

## Use Case: Return genre
Name: return_genre

What it does: Predicts the genre of a given mp3 file.  Returns a pie chart of the probabilites of nonzero genres.  

Inputs: song_mp3: _mp3_ of a song.

Outputs: pie_chart: _plt.pie_ chart displaying the probabilities of nonzero genres.
	 most_likely_genre: _string_ specifying the most likely genre. 

## Use case: Generate spectrogram from audio
Name: generate_spectrogram

What it does: Generates a spectrogram of an mp3 file. 

Inputs: song_mp3: _mp3_ of a song
	sample_rate: _int_ sample rate of the song
	n_fft: _int_ length of the fft window
	hop length: _int_ number of samples between successive frames

Outputs: spectrogram: spectrogram of audio file
	
## Use Case: Train Network
Name: train_network

What it does: Trains the neural network. Validates during the training process.

Inputs: train_data: data to train the network 
	validation_data: data to validate the network 
	model: architecture of network
	optimizer: optimizer to use
	loss function: loss function to use
	
Outputs: epoch: _int_ Epoch number
	 loss: _float_ current loss
	 
## Use Case: Test Network
Name: test_network

What it does: Tests the network. 

Inputs: test_data: data to test the network

Outputs: accuracy: _float_ test accuracy

## Use Case: Pass mp3 to have genres predicted through the network

Name: predict_genre

What it does? Passes the mp3 through the trained network and returns a genre prediction.

Inputs: spectrogram: Spectrogram of audio file

Returns: genre_prediction: List of predicted genres.

# Interactions to accomplish use cases

## Return song genre
1) User passes an mp3 into return_genre and specifies up to how many similar genres to return
2) System calls generate_spectrogram to generate a spectrogram.
3) System calls predict_genre to run the spectrogram through the pre-trained NN and return probabilites of each genre prediction.
4) return_genre returns a pie chart of the genre probabilities and a print out of the ranked list of most likely genres.

## Return most popular song in genre
1) User calls return_most_popular_song on a given genre.
2) System returns title of most popular song.
3) System calls return_mp3 on the song title to see if the sound file is in the database.  If it is not, nothing happens further.  If the song is in the database, system asks user if they want to hear the song.
4) If user wants to hear song, system calls play_mp3 to return an interactive playable track.

## Return similar genres
1) User calls return_most_similar_genres on a given genre.
2) System returns list of most similar genres.

## Train network
1) User defines their own network architecture and calls train_network to train the model.
2) User calls test_network to test the accuracy of the network. 

# Preliminary Plan:
1) Download with smallest music dataset and experiment with the data to better understand it. 
2) Test various preprocessing techniques as well as network architectures and begin writing train_network, test_network, predict_genre, and return_genre. 
3) Write functions that don't depend on the genre prediction: play_song, return_mp3, return_most_popular_song, and return_most_similar_genres.




