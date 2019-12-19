# CS433-Machine Learning Project 2

### Authors: Gerald Sula - Ridha Chahed - Walid Ben Naceur

Prerequisites:
We use some external libraries in this project which need to be installed manualy:

- xgboost:
	`pip install xgboost`
- surprise:
	`pip install surprise`
	or if you are on windows and the previous command does not work:
	`conda install -c conda-forge scikit-surprise`
	

1. Explanation of the main helper functions that you can find in utils.py

    create_csv_submission(ids, y_pred, name):

        Create a csv file with the format required by aicrowd
        
    preprocess(data):
        Convert a dataframe of indices of the given aicrowd input format to simple row/column dataframe by stripping down the 'r_' and 'c_' strings and removieng one from the index as to start from 0.
        
    trainset_from_surprise_to_df(trainset)
        Create a Dataframe starting from a Surprise data set containing the 'User', 'Movie', 'Rating' columns extracted from the input
    
    predict_on_model(algo):
        Use a single Surprise model to predict on the user/movie indices present on the sample_submission.csv. The predictions are rounded up to the closest integer
		
    	Returns:
        A list of (row/column) pairs and a list of the prediction in those indices
            
    predict_on_all_models_and_features_xgb(xgb_model,models, mf_sgd_pair, mf_als_pair, bl_global, bl_movie,bl_user, df_features):
        Use the provided models and augmented features to predict on the user/movie indices present on the sample_submission.csv and combine the result with the provided weights. The predictions are rounded up to the closest integer
		
		Returns:
        A list of (row/column) pairs and a list of the prediction in those indices
        
		
TODO: features::
    replace_word_by_key(tweets,voc):
        Use the provided models and augmented features to predict on the user/movie indices present on the sample_submission.csv and combine the result with the provided weights. 
    The predictions are computed in different ways depending on the model/feature.
    The predictions are rounded up to the closest integer
    
    padding(tweets,sequence_length):
        Add padding to each tweet so that they all have the same length.
        The tweets here are in numeric forms (one index per word).
        RETURN: list of tweets that have all the same length.

    process_tweet_test(tweet):
        taking a tweet, split it in word, remove the \n at the end and then remove all digits, float and punctuation from it. It also               transform the occurences of hahahaha , hahaha into the only haha.
        Remove the index at the start of each tweet.
        RETURN: a processed tweet.
        
    process_tweets_test(tweets):
        Taking a list of tweets, processed each tweet using the functions replace_words process_tweet.
        Return a list of processed tweets.
        RETURN: a list of processed tweets.
    
    create_csv_submission(ids, y_pred, name):
        Creates an output file in csv format for submission to kaggle
        Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
        
2. How to run the code

    To run the code you must first place the files data_train.csv and sampleSubmission.csv in a folder named 'data' on the same directory as the location of the run.py file
    Afterwards running the code is done my running:
	`python run.py`
    
    When the code has finished running a file named 'submissionBlendedXgbFull.csv' will be created on the same directory
    
	
TODO:
	
3. Explanation of the different models

    We have created different models during this project.
    Here we will quickly talk about them and the helper functions associated.
    
    model_parameters_cnn():
        Parameters for the 2 CNN models:
        num_filters: number of filter in the convolutional layers
        hidden_dims: dimensions of the dense layer
        batch_size: size of a batch for training
        num_epochs: number of epochs for our training
        filter_sizes: size of our filters in the convolutional layers
        RETURN: the number of filters, the number of hidden dimensions, the batch_size, the number of epochs and the sizes of the filters.
        
    preparation_RNN(y):
        Take the labels y and replace 0 by [0,1] and 1 by [1,0]
        RETURN: new labels in binary form.
    
    cnn_parallel_init(num_filters, hidden_dims, batch_size, num_epochs, filter_sizes, embedding_dim, sequence_length):
        Create the CNN model using parallel convolutional blocks.
        It has two identical blocks of convolutional layers, batchnormalization layers and maxpooling layers.
        Then it has a flatten layer and two dense layers.
        RETURN: the compiled model.
        
    cnn_sequential_init(num_filters, hidden_dims, batch_size, num_epochs, filter_sizes, embedding_dim, sequence_length):
        Create the sequential CNN model. 
        With one embedding layer, two convolutional layers followed by a batchnormalization layer and a maxpooling layer.
        This is repeated a second time with a smaller number of filters.
        It has then a flatten layer and three dense layer.
        RETURN: the compiled model.
        
    rnn_init(embedding_dim, sequence_length):
        Create the RNN model.
        With one embedding layer, an LSTM layer of size 250 and a dense layer.
        RETURN: the compiled model.
        
    cnn_rnn_init(embedding_dim, sequence_length):
        Create the combined CNN and RNN model0
        With one embedding layer, followed by a convolutional layer, a LSTM layer of size 250 and finally a dense layer.
        RETURN: the compiled model.
