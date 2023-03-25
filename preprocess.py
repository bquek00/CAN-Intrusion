from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np

def load_data(infile, use_IAT = False, NN = False, numerical = False, window = None, randomise_window=False, split=True):
    """
    Loads and formats the canbus data and splits it into an array of 
    train and tests subsets. 

    Args:
        infile (str): Path to the CSV file containing the data.
        use_IAT (bool): Whether to generate inter-arrival times (IAT) from the data. 
        NN (bool): Whether to preprocess the data for neural network models.
        numerical (bool): Whether to convert the subclasses to numerical. 
        window (int): Size of the  window used for RNN data preparation. 
        split (bool): Whether to split the data into train and test subsets. 


    Returns:
        Train and tests subsets
    """
    data = pd.read_csv(infile)

    data = data[data.SubClass != 'Replay']

    data.replace(to_replace=['Fuzzy'], value=['Fuzzing'], inplace = True)

    if window != None: 
        data.sort_values(["Timestamp"], ascending=True)

    if use_IAT:
        last_IAT = {}
        data["IAT"] = data.apply(generate_IAT, axis = 1, args=(last_IAT, data))

    data["Data"] = data["Data"].fillna(0) 
    
    if numerical:
        dftemp = data.copy()
        dftemp.replace(to_replace=['Normal', 'DOS', 'Fuzzing', 'Impersonation'],
            value=[0,1,2,3], inplace= True)
        y = dftemp['SubClass'].values
    else: 
        y = data["SubClass"].values

    data.drop(["Class", "SubClass"], axis=1, inplace = True)

    hex_to_dec = lambda x: int(x, 16)
    remove_space = lambda x:  x.replace(" ","") 
    

    data["Arbitration_ID"]= data["Arbitration_ID"].astype(str)
    data["Arbitration_ID"] =data["Arbitration_ID"].apply(hex_to_dec)

    data["Data"] = data["Data"].astype(str)
    data["Data"]= data["Data"].apply(remove_space)
    data["Data"]= data["Data"].apply(hex_to_dec)

    if not NN: 
        # Decided to scale differently for Neural Networks see trainer.py
        avg =data[["Data"]].mean(axis=0)
        scale_avg = lambda x: x/avg
        data["Data"] = data["Data"].apply(scale_avg)

    X = data.values

    if not split:
       return X, y

    data_length = data.shape[0]
    train_split = int(data_length * 0.8)

    if window != None:
        start = 0
        X_train, y_train = custom_ts_multi_data_prep(X, y, start, train_split, window)
        X_test, y_test= custom_ts_multi_data_prep(X, y, train_split, None, window)
        return  X_train, X_test, y_train, y_test
    
    return train_test_split(X, y, test_size=0.20,random_state=0)

def generate_IAT(row, last_IAT, df):
    """
    Generate the mean IAT given the row. 

    Args:
        row: The current row
        last_IAT: a dictionary containing a list of IATs where keys are ID
    
    Return:
        Mean IAT
    """
    Id = row['Arbitration_ID']
    timestamp = row["Timestamp"]

    seen = last_IAT.get(Id)

    if seen == None: # The first time a row with this Id appears
        row_num = row.name
        last_IAT[Id] = timestamp
        df.drop(row_num, inplace = True)
    else:
        return abs(seen - timestamp)

def custom_ts_multi_data_prep(dataset, target, start, end, window):
    """
    Format the dataset for time series analysis.

    Args:
        dataset: The dataset.
        target: Target variable.
        start: Start index.
        end: End index.
        window: The window size.

    Returns:
        New X and Y

    References:
        - Apress, Hands-On Time Series Analysis with Python,
          https://github.com/Apress/hands-on-time-series-analylsis-python/blob/master/Chapter%207/1.%20Bidirectional%20LSTM%20Multivariate%20Horizon%20Style.ipynb
    """
    X = []
    y = []
    start = start + window 
    if end is None:
        end = len(dataset)
        
    for i in range(start, end+1):
        indices = range(i-window, i) 
        X.append(dataset[indices])
        
        indicey = i -1
        y.append(target[indicey])
			
    return np.array(X), np.array(y)