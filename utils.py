import csv

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
    rs = [int(r[0][1:]) for r in df.index.str.split('_')] #Take the r part and remove the letter
    cs = [int(c[1][1:]) for c in df.index.str.split('_')] #Take the c part and remove the letter
    df['user'], df['item'] = rs, cs
    return df
    

def predict_on_model(algo):
    
    data=read_txt("data/sample_submission.csv")
    test_indices=predict_on_model_line(data[1:])

    preds=[]
    ids=[]
    for i,j in test_indices:
        ids.append("r{0}_c{1}".format(i+1,j+1))
        uid= j
        iid= i
        pred = algo.predict(uid,iid)

        preds.append(int(round(pred.est)))
    return ids, preds