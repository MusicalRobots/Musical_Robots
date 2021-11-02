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

# Components

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

## Use Case: Predict genre
Name: predict_genre

What it does: Predicts the genre of a given mp3 file.  Returns a pie chart of the probabilites of nonzero genres.  

Inputs: song_mp3: _mp3_ of a song.

Outputs: pie_chart: _plt.pie_ chart displaying the probabilities of nonzero genres.
	 most_likely_genre: _string_ specifying the most likely genre. 

## Generate spectrogram from audio
Name: generate_spectrogram

What it does: Generates a spectrogram of an mp3 file. 

Inputs: song_mp3: _mp3_ of a song
	sample_rate: _int_ sample rate of the song
	n_fft: _int_ length of the fft window
	hop length: _int_ number of samples between successive frames
	
## Train Network
Name: train_network

What it does: Trains the neural network. Validates during the training process.

Inputs: train_data: data to train the network 
	validation_data: data to validate the network 
	model: architecture of network
	optimizer: optimizer to use
	loss function: loss function to use
	
Outputs: epoch: _int_ Epoch number
	 loss: _float_ current loss
	 
## Test Network
Name: test_network

What it does: Tests the network. 

Inputs: test_data: data to test the network

Outputs: accuracy: _float_ test accuracy
	




