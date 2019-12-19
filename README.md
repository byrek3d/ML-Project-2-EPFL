

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
	

1. Explanation of the main helper functions that you can find in utils.py and features.py

    - create_csv_submission(ids, y_pred, name):

        Create a csv file with the format required by aicrowd
        
    - preprocess(data):
    
        Convert a dataframe of indices of the given aicrowd input format to simple row/column dataframe by stripping down the 'r_' and 'c_' strings and removieng one from the index as to start from 0.
        
    - trainset_from_surprise_to_df(trainset)
    
        Create a Dataframe starting from a Surprise data set containing the 'User', 'Movie', 'Rating' columns extracted from the input
    
    - predict_on_model(algo):
    
        Use a single Surprise model to predict on the user/movie indices present on the sample_submission.csv. The predictions are rounded up to the closest integer
		
    	Returns:
        A list of (row/column) pairs and a list of the prediction in those indices
            
    - predict_on_all_models_and_features_xgb(xgb_model,models, mf_sgd_pair, mf_als_pair, bl_global, bl_movie,bl_user, df_features):
    
        Use the provided models and augmented features to predict on the user/movie indices present on the sample_submission.csv and combine the result with the provided weights. The predictions are rounded up to the closest integer
		
		Returns:
        A list of (row/column) pairs and a list of the prediction in those indices
        
		

    - feature_augmentation(sparse_matrix,blending_trainset):

	     Global_Average : Average rating of all the ratings
      
	     User_Average : User's Average rating
     
	     Movie_Average : Average rating of this movie
     
	     Similar users rating of this movie (cosine similarity) :
	     SimUser1, SimUser2, SimUser3, SimUser4, SimUser5 ( top 5 similar 		users who rated that movie.. ). 
	     For each similar user need to find the rating that he put for that movie if not available put the average rating of that user as an estimator.
    
	     Similar movies rated by this user (cosine similarity):
	     SimMovie1, SimMovie2, SimMovie3, SimMovie4, SimMovie5 ( top 5 similar movies rated by this user.. )
	     For each similar movie we need to find the rating that the user has given to it if not available give the similar movie average rating.

	    Return:
        Two computed dataframes of the features added, one with the 'User', 'Movie' columns, and the other without
    
  
    
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
    
	

	
3. Different Models We Use

   

 - SVD
 - SVD++
 - KNN (User, Movie)
 - CoClustering
 - SlopeOne
 - Matrix factorisation with SGD
 - Matrix factorisation with ALS
 - Baseline (Global, User,Movie)
 - NMF
 
	 For Blending:
	 
	-XGBoost
	-Least Squares
	-Logistic Regression
	-Sequential quadratic programming

For more information and details about the models and how we use them, look at the report.
