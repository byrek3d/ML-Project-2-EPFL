# ## Sparse Matrix Training 


import pandas as pd
from scipy import sparse
import numpy as np


from sklearn.metrics.pairwise import cosine_similarity
    


def feature_augmentation(sparse_matrix,blending_trainset):
    """TODO
    """  

    print("The training matrix shape is : (user, movie) : ",sparse_matrix.shape)

    users, movies = sparse_matrix.shape
    elem = sparse_matrix.count_nonzero()

    print("Sparsity of the training matrix : {0} % ".format((1 - (elem / (users * movies))) * 100))


    # ## Rating's averages

    # ### Rating's average over all data



    global_average = sparse_matrix.sum() / sparse_matrix.count_nonzero()
    print("The average rating over all movies of trainset is : {0} ".format(global_average) )


    # ### Rating's average per user



    print("Computing the rating's average per user")

    user_mean = []   #contains the mean rating for user i at index i

    for user_index in range(users):
        
            # find the non-zero ratings for each user in the dataset
            ratings = sparse_matrix[user_index, :]
            nonzeros_ratings = ratings[ratings.nonzero()]
            
            # calculate the mean if the number of elements is not 0
            if nonzeros_ratings.shape[1] != 0:
                user_mean.append(nonzeros_ratings.mean())
            else:
                user_mean.append(0)


    # ### Rating's average per movie



    print("Computing the rating's average per movie")

    movie_mean = []   #contains the mean rating for movie j at index j

    for movie_index in range(movies):
        
            # find the non-zero ratings for each user in the dataset
            ratings = sparse_matrix[:, movie_index]
            nonzeros_ratings = ratings[ratings.nonzero()]
            
            # calculate the mean if the number of elements is not 0
            if nonzeros_ratings.shape[1] != 0:
                movie_mean.append(nonzeros_ratings.mean())
            else:
                movie_mean.append(0)


    # ### Similarity Matrix 



    # get the indices of  non zero rows(users) from our sparse matrix
    row_ind, col_ind = sparse_matrix.nonzero()

    row_ind = sorted(set(row_ind))   #to have unique values and sorted if needed  
    col_ind = sorted(set(col_ind))


    # #### User-User similarity 



    top = 5 
    print("Computing top",top,"similar user for each user")




    user_simil_matrix = []

    for row in row_ind: 
        # get the similarity row for this user with all other users
        simil = cosine_similarity(sparse_matrix.getrow(row), sparse_matrix).ravel()
        
        # get the index of the top 5 
        top_users = np.argsort((simil))[::-1][1:top+1]
        user_simil_matrix.append(top_users)


    # #### Movie-Movie similarity



    top = 5 
    print("Computing top",top,"similar movie for each movie")




    movie_simil_matrix = []

    for col in col_ind: 
        # get the similarity col for this movie with all other movies
        simil = cosine_similarity(sparse_matrix.getcol(col).T, sparse_matrix.T).ravel()
        # get the index of the top 5 
        top_movies = np.argsort((simil))[::-1][1:top+1]
        movie_simil_matrix.append(top_movies)


    # ### Featurizing the trainset

    # Global_Average : Average rating of all the ratings
    #  
    # User_Average : User's Average rating
    # 
    # Movie_Average : Average rating of this movie
    # 
    # Similar users rating of this movie:
    # SimUser1, SimUser2, SimUser3, SimUser4, SimUser5 ( top 5 similar users who rated that movie.. )
    # 
    # Similar movies rated by this user:
    # SimMovie1, SimMovie2, SimMovie3, SimMovie4, SimMovie5 ( top 5 similar movies rated by this user.. )



    row_ind, col_ind = sparse_matrix.nonzero()




    df_featured_data = pd.DataFrame({'User': row_ind, 'Movie' : col_ind, 'Grade' : sparse_matrix.data, 'Global_Average' : global_average })




    df_featured_data['User_Average'] = df_featured_data['User'].map(lambda x: user_mean[x])
    df_featured_data['Movie_Average'] = df_featured_data['Movie'].map(lambda x: movie_mean[x])


    # Get the indices of the similar users



    df_featured_data['SimUser1'] = df_featured_data['User'].map(lambda x: int(user_simil_matrix[x][0]))
    df_featured_data['SimUser2'] = df_featured_data['User'].map(lambda x: int(user_simil_matrix[x][1]))
    df_featured_data['SimUser3'] = df_featured_data['User'].map(lambda x: int(user_simil_matrix[x][2]))
    df_featured_data['SimUser4'] = df_featured_data['User'].map(lambda x: int(user_simil_matrix[x][3]))
    df_featured_data['SimUser5'] = df_featured_data['User'].map(lambda x: int(user_simil_matrix[x][4]))


    # For each similar user need to find the rating that he put for that movie if not available put the average rating of that user as an estimator. 



    def Userfunction1(row):
        if(sparse_matrix[row['SimUser1'],row['Movie']] == 0):
            return user_mean[int(row['SimUser1'])]
        else:
            return sparse_matrix[row['SimUser1'],row['Movie']]




    df_featured_data['SimUser1'] = df_featured_data.apply(Userfunction1,axis=1)




    def Userfunction2(row):
        if(sparse_matrix[row['SimUser2'],row['Movie']] == 0):
            return user_mean[int(row['SimUser2'])]
        else:
            return sparse_matrix[row['SimUser2'],row['Movie']]




    df_featured_data['SimUser2'] = df_featured_data.apply(Userfunction2,axis=1)




    def Userfunction3(row):
        if(sparse_matrix[row['SimUser3'],row['Movie']] == 0):
            return user_mean[int(row['SimUser3'])]
        else:
            return sparse_matrix[row['SimUser3'],row['Movie']]




    df_featured_data['SimUser3'] = df_featured_data.apply(Userfunction3,axis=1)




    def Userfunction4(row):
        if(sparse_matrix[row['SimUser4'],row['Movie']] == 0):
            return user_mean[int(row['SimUser4'])]
        else:
            return sparse_matrix[row['SimUser4'],row['Movie']]




    df_featured_data['SimUser4'] = df_featured_data.apply(Userfunction4,axis=1)




    def Userfunction5(row):
        if(sparse_matrix[row['SimUser5'],row['Movie']] == 0):
            return user_mean[int(row['SimUser5'])]
        else:
            return sparse_matrix[row['SimUser5'],row['Movie']]




    df_featured_data['SimUser5'] = df_featured_data.apply(Userfunction5,axis=1)


    # Get the indices of the similar movies



    df_featured_data['SimMovie1'] = df_featured_data['Movie'].map(lambda x: int(movie_simil_matrix[x][0]))
    df_featured_data['SimMovie2'] = df_featured_data['Movie'].map(lambda x: int(movie_simil_matrix[x][1]))
    df_featured_data['SimMovie3'] = df_featured_data['Movie'].map(lambda x: int(movie_simil_matrix[x][2]))
    df_featured_data['SimMovie4'] = df_featured_data['Movie'].map(lambda x: int(movie_simil_matrix[x][3]))
    df_featured_data['SimMovie5'] = df_featured_data['Movie'].map(lambda x: int(movie_simil_matrix[x][4]))


    # For each similar movie we need to find the rating that the user has given to it if not available give the similar movie average rating.  



    def Moviefunction1(row):
        if(sparse_matrix[row['User'],row['SimMovie1']] == 0):
            return movie_mean[int(row['SimMovie1'])]
        else:
            return sparse_matrix[row['User'],row['SimMovie1']]




    df_featured_data['SimMovie1'] = df_featured_data.apply(Moviefunction1,axis=1)




    def Moviefunction2(row):
        if(sparse_matrix[row['User'],row['SimMovie2']] == 0):
            return movie_mean[int(row['SimMovie2'])]
        else:
            return sparse_matrix[row['User'],row['SimMovie2']]




    df_featured_data['SimMovie2'] = df_featured_data.apply(Moviefunction2,axis=1)




    def Moviefunction3(row):
        if(sparse_matrix[row['User'],row['SimMovie3']] == 0):
            return movie_mean[int(row['SimMovie3'])]
        else:
            return sparse_matrix[row['User'],row['SimMovie3']]




    df_featured_data['SimMovie3'] = df_featured_data.apply(Moviefunction3,axis=1)




    def Moviefunction4(row):
        if(sparse_matrix[row['User'],row['SimMovie4']] == 0):
            return movie_mean[int(row['SimMovie4'])]
        else:
            return sparse_matrix[row['User'],row['SimMovie4']]




    df_featured_data['SimMovie4'] = df_featured_data.apply(Moviefunction4,axis=1)




    def Moviefunction5(row):
        if(sparse_matrix[row['User'],row['SimMovie5']] == 0):
            return movie_mean[int(row['SimMovie5'])]
        else:
            return sparse_matrix[row['User'],row['SimMovie5']]




    df_featured_data['SimMovie5'] = df_featured_data.apply(Moviefunction5,axis=1)


    # ### Featurizing the blending trainset



    df_blending_trainset=[]

    for u,m,r in blending_trainset:
        df_blending_trainset.append([u,m,r])
        
    df_featured_blending_trainset = pd.DataFrame(df_blending_trainset)
    df_featured_blending_trainset = df_featured_blending_trainset.rename({0:'User',1:'Movie',2:'Rating'},axis =1)




    df_featured_blending_trainset['User_Average'] = df_featured_blending_trainset['User'].map(lambda x: user_mean[x])
    df_featured_blending_trainset['Movie_Average'] = df_featured_blending_trainset['Movie'].map(lambda x: movie_mean[x])


    # Get the indices of the similar users



    df_featured_blending_trainset['SimUser1'] = df_featured_blending_trainset['User'].map(lambda x: int(user_simil_matrix[x][0]))
    df_featured_blending_trainset['SimUser2'] = df_featured_blending_trainset['User'].map(lambda x: int(user_simil_matrix[x][1]))
    df_featured_blending_trainset['SimUser3'] = df_featured_blending_trainset['User'].map(lambda x: int(user_simil_matrix[x][2]))
    df_featured_blending_trainset['SimUser4'] = df_featured_blending_trainset['User'].map(lambda x: int(user_simil_matrix[x][3]))
    df_featured_blending_trainset['SimUser5'] = df_featured_blending_trainset['User'].map(lambda x: int(user_simil_matrix[x][4]))


    # For each similar user need to find the rating that he put for that movie if not available put the average rating of that user as an estimator. 



    df_featured_blending_trainset['SimUser1'] = df_featured_blending_trainset.apply(Userfunction1,axis=1)
    df_featured_blending_trainset['SimUser2'] = df_featured_blending_trainset.apply(Userfunction2,axis=1)
    df_featured_blending_trainset['SimUser3'] = df_featured_blending_trainset.apply(Userfunction3,axis=1)
    df_featured_blending_trainset['SimUser4'] = df_featured_blending_trainset.apply(Userfunction4,axis=1)
    df_featured_blending_trainset['SimUser5'] = df_featured_blending_trainset.apply(Userfunction5,axis=1)




    df_featured_blending_trainset['SimMovie1'] = df_featured_blending_trainset['Movie'].map(lambda x: int(movie_simil_matrix[x][0]))
    df_featured_blending_trainset['SimMovie2'] = df_featured_blending_trainset['Movie'].map(lambda x: int(movie_simil_matrix[x][1]))
    df_featured_blending_trainset['SimMovie3'] = df_featured_blending_trainset['Movie'].map(lambda x: int(movie_simil_matrix[x][2]))
    df_featured_blending_trainset['SimMovie4'] = df_featured_blending_trainset['Movie'].map(lambda x: int(movie_simil_matrix[x][3]))
    df_featured_blending_trainset['SimMovie5'] = df_featured_blending_trainset['Movie'].map(lambda x: int(movie_simil_matrix[x][4]))


    # For each similar movie we need to find the rating that the user has given to it if not available give the similar movie average rating.  



    df_featured_blending_trainset['SimMovie1'] = df_featured_blending_trainset.apply(Moviefunction1,axis=1)
    df_featured_blending_trainset['SimMovie2'] = df_featured_blending_trainset.apply(Moviefunction2,axis=1)
    df_featured_blending_trainset['SimMovie3'] = df_featured_blending_trainset.apply(Moviefunction3,axis=1)
    df_featured_blending_trainset['SimMovie4'] = df_featured_blending_trainset.apply(Moviefunction4,axis=1)
    df_featured_blending_trainset['SimMovie5'] = df_featured_blending_trainset.apply(Moviefunction5,axis=1)




    df_featured_blending_trainset.SimUser1=df_featured_blending_trainset.SimUser1.astype(float)
    df_featured_blending_trainset.SimUser2=df_featured_blending_trainset.SimUser2.astype(float)
    df_featured_blending_trainset.SimUser3=df_featured_blending_trainset.SimUser3.astype(float)
    df_featured_blending_trainset.SimUser4=df_featured_blending_trainset.SimUser4.astype(float)
    df_featured_blending_trainset.SimUser5=df_featured_blending_trainset.SimUser5.astype(float)
    df_featured_blending_trainset.SimMovie1=df_featured_blending_trainset.SimMovie1.astype(float)
    df_featured_blending_trainset.SimMovie2=df_featured_blending_trainset.SimMovie2.astype(float)
    df_featured_blending_trainset.SimMovie3=df_featured_blending_trainset.SimMovie3.astype(float)
    df_featured_blending_trainset.SimMovie4=df_featured_blending_trainset.SimMovie4.astype(float)
    df_featured_blending_trainset.SimMovie5=df_featured_blending_trainset.SimMovie5.astype(float)




    df_featured_blending_trainset.drop(['Rating'],inplace=True, axis=1)




    #Must save a copy with the "User" and "Movie" columns to be used in predict_on_models
    df_featured_blending_trainset_no_user_movie=df_featured_blending_trainset.drop(['User','Movie'], axis=1)

    return df_featured_blending_trainset,df_featured_blending_trainset_no_user_movie