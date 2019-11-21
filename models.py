from surprise import BaselineOnly
from surprise import accuracy
from surprise import SlopeOne
from surprise import SVD
from surprise import KNNBasic


def baseline(trainset, testset): 
    model = BaseLineOnly()
    pred_train = model.fit_transform(trainset)
    pred_test = model.transform(testset)
    rmse = accuracy.rmse(pred_test)
    return pred_train, pred_test, rmse
    

def SlopeOne(trainset,testset):
    model = SlopeOne()
    pred_train = model.fit_transform(trainset)
    pred_test = model.transform(testset)
    rmse = accuracy.rmse(pred_test)
    return pred_train, pred_test, rmse

def SVD(trainset,testset,n_factors=20, n_epochs=40, lr_all=0.005, reg_all=0.02):
    model = SVD(n_factors, n_epochs, lr_all, reg_all)
    pred_train = model.fit_transform(trainset)
    pred_test = model.transform(testset)
    rmse = accuracy.rmse(pred_test)
    return pred_train, pred_test, rmse

def movie_knn(trainset, testset, fullset, labels, min_support=10, k=40):
    model_parameters = {
      'name': 'pearson',
      'user_based': False,
      'min_support': min_support  #minimum number of common use/item to be compared. 
    }
    model = KNNBasic(k,sim_options=model_parameters)
    pred_train = model.fit_transform(trainset)
    pred_test = model.transform(testset)
    rmse = accuracy.rmse(pred_test)
    return pred_train, pred_test, rmse
    
def user_knn(trainset, testset, fullset, labels, min_support=10, k=40):
    model_parameters = {
      'name': 'pearson',
      'user_based': True,
      'min_support': min_support  #minimum number of common use/item to be compared. 
    }
    model = KNNBasic(k,sim_options=model_parameters)
    pred_train = model.fit_transform(trainset)
    pred_test = model.transform(testset)
    rmse = accuracy.rmse(pred_test)
    return pred_train, pred_test, rmse

def co_clustering(trainset, testset, fullset, labels, n_clstr_usr=3, n_clstr_mv=3):
    model = CoClustering(n_cltr_u=n_clstr_usr, n_cltr_i=n_clstr_mv)
    pred_train = model.fit_transform(trainset)
    pred_test = model.transform(testset)
    rmse = accuracy.rmse(pred_test)
    return pred_train, pred_test, rmse
