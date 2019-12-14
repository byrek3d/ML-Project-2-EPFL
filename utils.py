import csv
import numpy as np 
import pandas as pd

def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()
    
    
def deal_line(line):
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
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
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
    """ Preproccess pandas dataframes such as training set and validation set
    :return: df a with same index and fields item, user and rating
    :param data: Dataframe with index 'r_X_c_Y' where X is the user and Y the movie e.g. r44_c1, and value the rating
    :type data: pandas.DataFrame
    Examples:
        training = preprocess(the_data)
        val_set = preprocess(validation_set)
    """
    
    df = data.copy()
    # process removing the r or c and converting them into integers
    # Remove 1 from the received indexes as to have the matrices start at index 0
    rs = [int(r[0][1:])-1 for r in df.index.str.split('_')] #Take the r part and remove the letter
    cs = [int(c[1][1:])-1 for c in df.index.str.split('_')] #Take the c part and remove the letter
    df['user'], df['item'] = rs, cs 
    return df
    
def read_file():
   
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
#             print("est:",m.predict(uid,iid).est)
#             print("w:",w)
            pred = pred+m.predict(uid,iid).est*w
#         print("---Res:",pred)

        preds.append(int(round(pred)))
    return ids, preds
def predict_on_models2(models, weights):
    
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
            
        df_models = pd.DataFrame(np.reshape(model_preds, (1,-1)), columns = ['dfCC', 'dfBL', 'dfSVD', 'dfSVDpp', 'dfNMF', 'dfKNNMovie', 'dfKNNUser','dfSO'] )
        res=xgb_model.predict(df_models)
        preds.append(res)
        
        ##TODO, may want to do the rounding here
    return ids, preds
def predict_on_models_xgb(models, df_features , xgb_model):
    
    
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