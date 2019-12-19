1. Explanation of the helper functions that you can find in implementations.py

    process_tweet(tweet):
        Given a tweet, process it such that you remove the \n at the end, all digits, float and punctuation from it. 
        Also transform hahahaha strings of size > 4 into haha.
        RETURN: a processed tweet
        
    create_sets():
        Create sets of meanings: for given words having the same meaning in the train data, 
        place them in a set from all which all the words will be replaced by the same representant. 
        We have created these sets looking a the occurences of a word in the negative and positive train data. We have:
        - set_laugh
        - set_good
        - set_happy
        - set_love
        - set_sad
        - set_angry
        - set_bad
        - set_neutral
        RETURN: the different sets
        
    replace_words(tweet):
        Replace all the occurences of a word contained in one of the given sets by the representant of this set.
        The set neutral exists if you want to speed up the computations a lot, but you will loose some accuracy afterwards.
        RETURN: tweet with words contained in a set replaced.
    
    process_tweets(tweets):
        Taking a list of tweets, processed each tweet using the functions replace_words process_tweet.
        RETURN: a list of processed tweets.
    
    compute_error(data, user_features, item_features, cooc_row,cooc_col):
        Compute the loss (RMSE) of the prediction of nonzero elements.
        RETURN: the value of the RMSE.
            
    remove_n(data):
        Remove the occurences of the \n at the end of a line when opening a file.
        This function is used when loading already processed data so that we don't have to do the all processing again.
        RETURN: the data without the \n at the end of each line.
        
    replace_word_by_key(tweets,voc):
        For each tweet, replace each words by the corresponding index in the vocabulary map.
        For example replace 'the' by the index 1.
        It drops the words that are not in the vocabulary.
        RETURN: the final tweet list containing tweets with a numeric representation.
    
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

    Since we are using the code given for creating the cooc matrix, we can't do everything in a simple run.py.
    So to run our code, you have to do the following steps in order:
        1) run process_tweets.py to create new files containing processed tweets so that you can construct the vocabulary we want.
        2) use build_vocab.sh where you replace the name of the file from which you take the data by:
                'processed_pos_tweets_fact.txt' 'processed_neg_tweets_fact.txt'
        3) run cut_vocab.sh
        4) run python3 pickle_vocab.py
        5) run python3 cooc.py
        6) run the run.py
    
    You can access all the different models we tried in the file models.py.
    Our run contain only the best model we had (the one giving the result on the submission website).
    
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