import csv
import numpy as np 
import pandas as pd

def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()
    
    
def deal_line(line):
     """
    Return the index of the row and column form a csv line, by also removing 1 (so that the indices start at 0
    ----------
    
    line : String
        A line of th format of the provided csv file
    Returns:
        The index of the row and column subtracted by 1
    """
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row)-1, int(col)-1 #remove one to have same indices
    
    
def predict_on_model_line(data):
    """preprocessing the text data, conversion to numerical array format."""
    data = [deal_line(line) for line in data]
 
    return data


def create_csv_submission(ids, y_pred, name):
     """
    Create a csv with the format required by aicrowd
    ----------
    ids : list
        List of pairs of (row,column) indices
    y_pred : list
        List of prediction of the rating
    name : String
        The name of the csv file to be created
   
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        if(1176952!=len(list(zip(ids, y_pred)))):
            print("Error! Missmatch in the lengs of the csv rows")
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': r1,'Prediction': r2})

def preprocess(data):
     """
    Convert a dataframe of indices of the given aicrowd input format to simple row/column dataframe
    ----------
    
    data : Dataframe
        The input in aicrowd format
    Returns:
        A dataframe of user/item indices
    """
    
    df = data.copy()
    # process removing the r or c and converting them into integers
    # Remove 1 from the received indexes as to have the matrices start at index 0
    rs = [int(r[0][1:])-1 for r in df.index.str.split('_')] #Take the r part and remove the letter
    cs = [int(c[1][1:])-1 for c in df.index.str.split('_')] #Take the c part and remove the letter
    df['user'], df['item'] = rs, cs 
    return df
    
def read_file():
     """
    Read the sample_submission.csv file and extract the information into 3 lists
    ----------
    
   
    Returns:
        A list containing the lines of the csv file, a list containing the rwo index and a list containing the column index
        
   """
    data=read_txt("data/sample_submission.csv")
    test_indices=predict_on_model_line(data[1:])

    uid=[]
    iid=[]
    ids = []
    
    for i,j in test_indices:
        ids.append("r{0}_c{1}".format(i+1,j+1))
        uid.append(i)
        iid.append(j)

    return  ids,uid,iid

def predict_on_model(algo):
    """
    Use the provided model to predict on the user/movie indices present on the sample_submission.csv. The predictions are rounded up to the closest integer
    ----------
    
    algo: surprise.prediction_algorithms
       The model to be used in the prediction
       
    Returns:
        A list of (row/column) pairs and a list of the prediction in those indices
        
   """
    data=read_txt("data/sample_submission.csv")
    test_indices=predict_on_model_line(data[1:])

    preds=[]
    ids=[]
    acc=0
    for i,j in test_indices:
        acc=acc+1
        if (acc%100000==0):
            print(acc)
        ids.append("r{0}_c{1}".format(i+1,j+1))
        uid= i
        iid= j
        pred = algo.predict(uid,iid)

        preds.append(int(round(pred.est)))
    return ids, preds

def predict_on_models(models, weights):
    """
    Use the provided models to predict on the user/movie indices present on the sample_submission.csv and combine the result with the provided weights. The predictions are rounded up to the closest integer
    ----------
    
    models: list
       List of models to be used in the prediction
    weights: list
        List of weights corresponding to each of the provided models
       
    Returns:
        A list of (row/column) pairs and a list of the prediction in those indices
    """       
    zippedMW=list(zip(models,weights))
    data=read_txt("data/sample_submission.csv")
    test_indices=predict_on_model_line(data[1:])

    
    preds=[]
    ids=[]
    acc=0
    for i,j in test_indices:
        acc=acc+1
        if (acc%10000==0):
            print(acc)
        ids.append("r{0}_c{1}".format(i+1,j+1))
        uid= i
        iid= j
        pred=0
        for m, w in zippedMW:

            pred = pred+m.predict(uid,iid).est*w


        preds.append(int(round(pred)))
    return ids, preds

def predict_on_models2(models, weights):
    """
    Use the provided models to predict on the user/movie indices present on the sample_submission.csv and combine the result with the provided weights. The predictions are rounded up to the closest integer
    ----------
    
    models: list
       List of models to be used in the prediction
    weights: list
        List of weights corresponding to each of the provided models
       
    Returns:
        A list of (row/column) pairs and a list of the prediction in those indices
    """
    
    data=read_txt("data/sample_submission.csv")
    test_indices=predict_on_model_line(data[1:])

    preds=[]
    ids=[]
    for i,j in test_indices:
        ids.append("r{0}_c{1}".format(i+1,j+1))
        uid= i
        iid= j
        pred_list=[]
        
        acc=0
        for wclass in weights:
            acc=acc+1
            if (acc%1000==0):
                print(acc)
            zippedMW=list(zip(models,wclass))
            pred = 0
            for m,w in zippedMW:
#                 print("m:",m)
#                 print("w:",w)
                pred = pred+m.predict(uid,iid).est*w
                
            pred = np.exp(pred) / (1 + np.exp(pred) )   
            pred_list.append(pred) 
#         print(pred_list)

        preds.append(np.argmax(np.array(pred_list))+1)
    return ids, preds

def predict_on_models_xgb_old(models, xgb_model):
    """
    Use the provided models to predict on the user/movie indices present on the sample_submission.csv. The predictions are weighted using the provided xgboost model. The predictions are rounded up to the closest integer
    ----------
    
    models: list
       List of models to be used in the prediction
    xgb_model: xgboost
       A xgboost ensemble model trained on the provided prediction models
    Returns:
        A list of (row/column) pairs and a list of the prediction in those indices
    """
    
    data=read_txt("data/sample_submission.csv")
    test_indices=predict_on_model_line(data[1:])

    
    preds=[]
    ids=[]
    acc=0
    for i,j in test_indices:
        acc=acc+1
        if (acc%100000==0):
            print(acc)
        ids.append("r{0}_c{1}".format(i+1,j+1))
        uid= i
        iid= j
        model_preds=[]
        for m in models:

            model_preds.append(m.predict(uid,iid).est)# TODO: maybe round here
            
        df_models = pd.DataFrame(np.reshape(model_preds, (1,-1)), columns = ['dfCC', 'dfSVDpp','dfKNNMovie', 'dfKNNUser'] )
        res=xgb_model.predict(df_models)
        preds.append(res)
        
        ##TODO, may want to do the rounding here
    return ids, preds


def predict_on_models_xgb(models, df_features , xgb_model):
    
    """
    Use the provided models and augmented features to predict on the user/movie indices present on the sample_submission.csv and combine the result with the provided weights. The predictions are rounded up to the closest integer
    ----------
    
    models: list
       List of models to be used in the prediction
    df_features: Dataframe
        Dataframe containing additional features on the data 
    xgb_model: xgboost
       A xgboost ensemble model trained on the provided prediction models
       
    Returns:
        A list of (row/column) pairs and a list of the prediction in those indices
    """
    data=read_txt("data/sample_submission.csv")
    test_indices=predict_on_model_line(data[1:])

    
    preds=[]
    ids=[]
    for i,j in test_indices:
        ids.append("r{0}_c{1}".format(i+1,j+1))
        uid= i
        iid= j
        model_preds=[]
        for m in models:

            model_preds.append(m.predict(uid,iid).est)# TODO: maybe round here

            
        df_models = pd.DataFrame(np.reshape(model_preds, (1,-1)), columns = ['dfCC', 'dfBL', 'dfSVD', 'dfSVDpp', 'dfNMF', 'dfKNNMovie', 'dfKNNUser','dfSO'] )
        
        features_row = df_features[(df_features.User == uid) & (df_features.Movie == iid)].drop(['User','Movie'],axis = 1 )
        
        df_merged=pd.concat([df_models, features_row], axis=1)
        res=xgb_model.predict(df_merged)
        preds.append(res)
        
        ##TODO, may want to do the rounding here
    return ids, preds