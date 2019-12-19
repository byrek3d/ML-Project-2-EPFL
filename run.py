
# # Netflix Recommender System
# # Project 2 of Machine Learning course CS-433 at EPFL
# # Authors: Gerald Sula - Ridha Chahed - Walid Ben Naceur

# Load libraries

import numpy as np
import seaborn as sns 
import pandas as pd

from scipy import sparse
from scipy.sparse import csr_matrix

import scipy
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt


import models as m
import utils as u
import features as f

from surprise.dataset import * 
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise import accuracy
from surprise.model_selection import GridSearchCV
from surprise import BaselineOnly,CoClustering,SVD,SVDpp,NMF,SlopeOne,KNNBasic

import xgboost as xgb


#Optimal parameters for the models we use

n_factorsSVD=80
n_epochsSVD=800
lr_allSVD=0.001667
reg_allSVD=0.1

epochs_SVDpp= 30

n_cltr_uCC=13
n_cltr_iCC=13
n_epochsCC=200

n_factorsNMF= 15
n_epochsNMF= 50
reg_puNMF=0.06
reg_qiNMF= 0.06
reg_buNMF= 0.02
reg_biNMF=0.06

bsl_options= {'method': 'als', 'n_epochs': 10, 'reg_u': 15, 'reg_i': 5}

model_parameters_user = {
      'name': 'pearson',
      'user_based': True
    }

k_user=100

model_parameters_movie = {
      'name': 'pearson',
      'user_based': False
    }

k_movie=300

gammaSGD = 0.025
num_featuresSGD = 20   
lambda_userSGD = 0.1
lambda_itemSGD = 0.01
num_epochsSGD = 20   

num_featuresALS = 20   
lambda_userALS = 0.081
lambda_itemALS = 0.081
stop_criterionALS = 1e-5

#---------------------------------------------------------------------------------------------
# Load data
print("-Loading the data")
raw_data = pd.read_csv('data/data_train.csv', header=0, index_col=0, names=['Id', 'rating'])
data = u.preprocess(raw_data).reset_index().drop(columns=['Id'])




print("Total data\n")
print("-"*50)
print("Total no of ratings :",data.shape[0])
print("Total No of Users   :", len(np.unique(data.user)))
print("Total No of movies  :", len(np.unique(data.item)))



# Suprise Reader

reader=Reader(rating_scale=(1.0,5.0))
formatted_data= Dataset.load_from_df(data[['user','item','rating']],reader)


#Split the data into trainset and blending_trainset
print("Seperating the data in 2 datasets: one for training the models and one for training the blending model:")
trainset, blending_trainset = train_test_split(formatted_data, test_size=.2 ,random_state=1)




df_trainset = u.trainset_from_surprise_to_df(trainset)
ratings = sparse.csr_matrix((df_trainset.Rating.values, (df_trainset.Movie.values,df_trainset.User.values)))
print("The training matrix shape is : (movie, user) : ",ratings.shape)


#Split the trainset into train and test to be used for the models we define
train, test = m.split_data(ratings, p_test=0.1)


#Save the label of the second dataset in a dataframe
label_blending_trainset = []

for a,b,c in blending_trainset:
    label_blending_trainset.append(c)

df_label_blending_trainset=pd.DataFrame(label_blending_trainset)



#Training the different models

print("-Training CoCluster")
algoCC= CoClustering(n_cltr_i=n_cltr_iCC, n_cltr_u=n_cltr_uCC, n_epochs=n_epochsCC)
algoCC.fit(trainset)

print("-Training Baseline")
algoBL=BaselineOnly(bsl_options=bsl_options)
algoBL.fit(trainset)

print("-Training SVD")
algoSVD=SVD( n_factors=n_factorsSVD, n_epochs=n_epochsSVD, lr_all=lr_allSVD,reg_all=reg_allSVD)
algoSVD.fit(trainset)

print("-Training SVD++")
algoSVDpp = SVDpp(n_factors=n_factorsSVD, n_epochs=epochs_SVDpp, lr_all=lr_allSVD,reg_all=reg_allSVD)
algoSVDpp.fit(trainset)

print("-Training NMF")
algoNMF = NMF(n_factors=n_factorsNMF, n_epochs=n_epochsNMF, reg_pu=reg_puNMF, reg_qi=reg_qiNMF, reg_bu=reg_buNMF, reg_bi=reg_biNMF)
algoNMF.fit(trainset)

print("-Training KNN on movie")
algoKNNMovie =KNNBasic(model_parameters=model_parameters_movie, k=k_movie)
algoKNNMovie.fit(trainset)

print("-Training KNN on user")
algoKNNUser =KNNBasic(model_parameters=model_parameters_user,k=k_user)
algoKNNUser.fit(trainset)

print("-Training Slope One")
algoSO = SlopeOne()
algoSO.fit(trainset)

print("-Training MF SGD")
user_sgd, movie_sgd = m.matrix_factorization_SGD(train, test,gammaSGD, num_featuresSGD,lambda_userSGD,lambda_itemSGD,num_epochsSGD)

print("-Training MF ALS")
user_als,movie_als = m.ALS(train, test,num_featuresALS,lambda_userALS, lambda_itemALS,stop_criterionALS)

print("-Training global baseline")
baseline_global=m.baseline_global_mean(train, test)

print("-Training user baseline")
baseline_user=m.baseline_user_mean(train, test)

print("-Training movie baseline")
baseline_movie=m.baseline_movie_mean(train, test)


#Predicting on the blending_trainset using the previously trained models and saving the results as dataframes

print("-For the Blending algorithm, we predict on the second dataset using the trained models")
predCC=algoCC.test(blending_trainset)
dfCC=u.pred_from_suprise_to_df(predCC)

predBL=algoBL.test(blending_trainset)
dfBL=u.pred_from_suprise_to_df(predBL)

predSVD=algoSVD.test(blending_trainset)
dfSVD=u.pred_from_suprise_to_df(predSVD)

predSVDpp=algoSVDpp.test(blending_trainset)
dfSVDpp=u.pred_from_suprise_to_df(predSVDpp)

predNMF=algoNMF.test(blending_trainset)
dfNMF=u.pred_from_suprise_to_df(predNMF)

predKNNMovie=algoKNNMovie.test(blending_trainset)
dfKNNMovie=u.pred_from_suprise_to_df(predKNNMovie)

predKNNUser=algoKNNUser.test(blending_trainset)
dfKNNUser=u.pred_from_suprise_to_df(predKNNUser)

predSO=algoSO.test(blending_trainset)
dfSO=u.pred_from_suprise_to_df(predSO)


dfMFSGD=[]

for uid,iid,_ in blending_trainset: #(row,col) => (user,movie)

    user_data = user_sgd[:,uid]  
    movie_data = movie_sgd[:,iid]
            
    prediciton= movie_data @ user_data.T
    
    dfMFSGD.append(prediciton)
dfMFSGD=pd.DataFrame([dfMFSGD]).transpose()



dfMFALS=[]

for uid,iid,_ in blending_trainset: #(row,col) => (user,movie)

    user_data = user_als[:,uid]  
    movie_data = movie_als[:,iid]
            
    prediciton= movie_data @ user_data.T
    
    dfMFALS.append(prediciton)
dfMFALS=pd.DataFrame([dfMFALS]).transpose()



dfBLGlobal = pd.DataFrame( [baseline_global] *len(blending_trainset) )



dfBLUser = []
for user,movie,_ in blending_trainset:
    dfBLUser.append(baseline_user[0,user])
dfBLUser = pd.DataFrame(dfBLUser)



dfBLMovie = []
for user,movie,_ in blending_trainset:
    dfBLMovie.append(baseline_movie [movie,0])
dfBLMovie = pd.DataFrame(dfBLMovie)  



print("-Starting Feature augmntation\n")
print("-"*50)

sparse_matrix = sparse.csr_matrix((df_trainset.Rating.values, (df_trainset.User.values,df_trainset.Movie.values)))

#Perform feature augmentation on the trainset
df_featured_blending_trainset,df_featured_blending_trainset_no_user_movie=f.feature_augmentation(sparse_matrix,blending_trainset)


print("-Starting Blending with xgboost\n")
print("-"*50)


df_val=pd.concat([dfCC,dfBL,dfSVD,dfSVDpp,dfNMF,dfKNNMovie,dfKNNUser,dfSO,dfMFSGD,dfMFALS,dfBLGlobal,dfBLMovie, dfBLUser],ignore_index=True,axis=1)
df_val=df_val.rename({0:'dfCC',1:'dfBL',2:'dfSVD',3:'dfSVDpp',4:'dfNMF',5:'dfKNNMovie',6:'dfKNNUser',7:'dfSO',8:'dfMFSGD',9:'dfMFALS',10:'dfBLGlobal',11:'dfBLMovie',12:'dfBLUser'},axis=1)


df_val=pd.concat([df_val,df_featured_blending_trainset_no_user_movie],axis=1)


print("-Training the xgb model using 25 models and added features")
model_xgb= xgb.XGBRegressor(silent=True, n_jobs=25, random_state=1,n_estimators=100)

model_xgb.fit(df_val,label_blending_trainset, eval_metric='rmse')


print("-Predicting on the models using the computed importance weights")

ids, preds = u.predict_on_all_models_and_features_xgb(model_xgb,[algoCC, algoBL,algoSVD, algoSVDpp,algoNMF,algoKNNMovie,algoKNNUser,algoSO],
                                          [user_sgd, movie_sgd],[user_als, movie_als],
                                          baseline_global,baseline_movie,baseline_user,df_featured_blending_trainset)





    
u.create_csv_submission(ids, preds, "submissionBlendedXgbFull.csv")

print("Csv file succesfuly created")

