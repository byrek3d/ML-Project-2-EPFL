# # In this file we define some models to use for the prediction that were not available in any libraries. 
# #These are based on the lab 10 of the course.

import numpy as np
import scipy.sparse as sp
import random, copy


def split_data(ratings, p_test=0.1, verbose=False):
    """split the ratings to training data and test data.
    Args:
        ratings: 
            the data to be split
        p_test:
            the percentage of data to be used for the test set
    Returns:
        Train set and test set
    """

    num_rows, num_cols=ratings.shape
    train = sp.lil_matrix((num_rows, num_cols))
    test = sp.lil_matrix((num_rows, num_cols))
    
    
    nz_items, nz_users = ratings.nonzero()
    for user in set(nz_users):
        row, col = ratings[:, user].nonzero()
        selects = np.random.choice(row, size=int(len(row) * p_test))
        residual = list(set(row) - set(selects))

        # add to train set
        for r in residual:      
            train[r, user] = ratings[r, user]

        # add to test set
        for s in selects:
            test[s, user] = ratings[s, user]
    if(verbose==True):
        print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
        print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
    return  train, test

def init_MF(train, num_features):
    """init the parameter for matrix factorization."""

    num_items, num_users = train.shape
    user_features = np.random.rand(num_features,num_users)/num_users
    user_features[0,:]=np.ones((num_users,))
    item_features = np.random.rand(num_features,num_items)/num_items
    item_features[0,:]=sp.csr_matrix.mean(train,axis=1).reshape(num_items,)

    return user_features, item_features

def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    mse=0
    for row,col in nz:
        w_d = item_features[:,row]
        z_n = user_features[:,col]
        prediction= w_d @ z_n.T
        error_prediction = (data[row,col] - prediction ) **2 
        mse+=error_prediction
    
    return np.sqrt(mse / len(nz))

def build_index_groups(train):
    """build groups for nnz rows and cols."""
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    grouped_nz_train_byrow = group_by(nz_train, index=0)
    nz_row_colindices = [(g, np.array([v[1] for v in value]))
                         for g, value in grouped_nz_train_byrow]

    grouped_nz_train_bycol = group_by(nz_train, index=1)
    nz_col_rowindices = [(g, np.array([v[0] for v in value]))
                         for g, value in grouped_nz_train_bycol]
    return nz_train, nz_row_colindices, nz_col_rowindices

def baseline_global_mean(train, test, verbose=False):
    """Compute the baseline global mean of the train data
    Args:
        train: 
            the data to be used as input
        test:
            the data to be used for measuring the rmse
    Returns:
        The global mean of the train data
    """
    
    global_mean_train = train[train.nonzero()].mean()
    
    test_nonzero_dense = test[test.nonzero()].todense()
    
    mse = calculate_mse( test_nonzero_dense, global_mean_train )
    
    rmse = np.sqrt( mse / test_nonzero_dense.shape[1] )
    
    if(verbose==True):
        print("Baseline global RMSE on test: ", rmse[0,0])
    
    return global_mean_train

def calculate_mse(real_label, prediction):
    """calculate MSE."""
    t = real_label - prediction
    return 1.0 * t.dot(t.T)

def baseline_user_mean(train, test,verbose=False):
    """Compute the baseline user mean of the train data
    Args:
        train: 
            the data to be used as input
        test:
            the data to be used for measuring the rmse
    Returns:
        The user mean of the train data
    """   
    mse = 0
    count = 0 
    num_items, num_users = train.shape
    
    sums=train.sum(axis=0) #sum for each user
    
    mean_user=np.zeros((1,num_users))

    for j in range(0,num_users):
        if(sums[0,j] != 0):
            elems = train[:,j]
            elems_nonzero = elems[elems.nonzero()]
            mean_user[0,j] = elems_nonzero.mean()
    
        for i in range(test.shape[0]):
            if(test[i,j] != 0):
                mean_user_elem = mean_user[0,j]
                mse += (test[i,j]-mean_user_elem )**2
                count+= 1
    rmse = np.sqrt( mse / count )
    if(verbose==True):
        print("Baseline User RMSE on test: ",rmse)
    return mean_user

def baseline_movie_mean(train, test, verbose=False):
    """Compute the baseline movie mean of the train data
    Args:
        train: 
            the data to be used as input
        test:
            the data to be used for measuring the rmse
    Returns:
        The movie mean of the train data
    """   
    mse = 0
    count = 0 
    num_items, num_users = train.shape
    
    sums=train.sum(axis=1) #sum for each user
    
    mean_item=np.zeros((num_items,1))

    for i in range(0,num_items):
        if(sums[i,0] != 0):
            elems = train[i,:]
            elems_nonzero = elems[elems.nonzero()]
            mean_item[i,0] = elems_nonzero.mean()
    
        for j in range(test.shape[1]):
            if(test[i,j] != 0):
                mean_item_elem = mean_item[i,0]
                mse += (test[i,j]-mean_item_elem)**2
                count+= 1
   
    rmse = np.sqrt( mse / count )
    if(verbose==True):
        print("Baseline Movie RMSE on test: ",rmse)
    return mean_item

def matrix_factorization_SGD(train, test,gamma, num_features,lambda_user,lambda_item,num_epochs,reg=True,verbose=False):
    """TODO
    """   
  
    errors = [0]
    

    # init matrix
    user_features, item_features = init_MF(train, num_features)   #Z0.T,W0
    
    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    print("learn the matrix factorization using SGD...")
    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        gamma /= 1.2
        
        for d, n in nz_train:
            
            item_data = item_features[:,d]  
            user_data = user_features[:,n]
            prediction = item_data @ user_data.T
    
            prediciton_error = train[d, n] - item_data @ user_data.T
        
            #compute derivative wrt w
            grad_w = -prediciton_error * user_data  
                
            #compute derivative wrt z 
            grad_z = -prediciton_error * item_data
   
            #update 
            if(reg):  
                item_features[:,d]-= gamma * ( grad_w + lambda_item * item_data)
                user_features[:,n]-= gamma * ( grad_z + lambda_user * user_data)
            
            else:
                item_features[:,d]-= gamma * grad_w
                user_features[:,n]-= gamma * grad_z
        
        rmse = compute_error(train, user_features, item_features, nz_train)
        if(verbose==True):
            print("iter: {}, RMSE on training set: {}.".format(it, rmse))
        
        errors.append(rmse)

    rmse = compute_error(test, user_features, item_features, nz_test)
    if(verbose==True):  
        print("RMSE on test data: {}.".format(rmse))
    
    return user_features, item_features


def update_user_feature(
        train, item_features, lambda_user,
        nnz_items_per_user, nz_user_itemindices):
    """update user feature matrix."""
    """the best lambda is assumed to be nnz_items_per_user[user] * lambda_user"""
    
    
    num_user = nnz_items_per_user.shape[0]
    num_feature = item_features.shape[0]
    lambda_I = lambda_user * np.eye(num_feature)
    updated_user_features = np.zeros((num_feature, num_user))

    for user, items in nz_user_itemindices:
        # extract the columns corresponding to the prediction for given item
        M = item_features[:, items]
        
        # update column row of user features
        V = M @ train[items, user]
        A = M @ M.T + nnz_items_per_user[user] * lambda_I
        X = np.linalg.solve(A, V)
        updated_user_features[:, user] = np.copy(X.T)
    return updated_user_features

def update_item_feature(
        train, user_features, lambda_item,
        nnz_users_per_item, nz_item_userindices):
    """update item feature matrix."""
    """the best lambda is assumed to be nnz_items_per_item[item] * lambda_item"""
    num_item = nnz_users_per_item.shape[0]
    num_feature = user_features.shape[0]
    lambda_I = lambda_item * sp.eye(num_feature)
    updated_item_features = np.zeros((num_feature, num_item))

    for item, users in nz_item_userindices:
        # extract the columns corresponding to the prediction for given user
        M = user_features[:, users]
        V = M @ train[item, users].T
        A = M @ M.T + nnz_users_per_item[item] * lambda_I
        X = np.linalg.solve(A, V)
        updated_item_features[:, item] = np.copy(X.T)
    return updated_item_features

def ALS(train, test,num_features,lambda_user, lambda_item,stop_criterion,verbose=False):
    """TODO
    """  

    change = 1
    error_list = [0, 0]
    

    # init ALS
    user_features, item_features = init_MF(train, num_features)
    
    # get the number of non-zero ratings for each user and item
    nnz_items_per_user, nnz_users_per_item = train.getnnz(axis=0), train.getnnz(axis=1)

    # group the indices by row or column index
    nz_train, nz_item_userindices, nz_user_itemindices = build_index_groups(train)

    # run ALS
    print("\nstart the ALS algorithm...")
    while change > stop_criterion:
        # update user feature & item feature
        user_features = update_user_feature(
            train, item_features, lambda_user,
            nnz_items_per_user, nz_user_itemindices)
        item_features = update_item_feature(
            train, user_features, lambda_item,
            nnz_users_per_item, nz_item_userindices)

        error = compute_error(train, user_features, item_features, nz_train)

        if(verbose==True):  
            print("RMSE on training set: {}.".format(error))

        error_list.append(error)
        change = np.fabs(error_list[-1] - error_list[-2])

    # evaluate the test error
    nnz_row, nnz_col = test.nonzero()
    nnz_test = list(zip(nnz_row, nnz_col))
    rmse = compute_error(test, user_features, item_features, nnz_test)
    if(verbose==True):  
        print("test RMSE after running ALS: {v}.".format(v=rmse))
    return user_features, item_features