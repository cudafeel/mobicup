import numpy as np
import re
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing

def dealDataset(r_filename,w_filename):
    train = pd.read_csv(r_filename)
    train.drop(['orderid', 'bikeid', 'biketype'], axis=1, inplace=True)
    temp = pd.DatetimeIndex(train['starttime'])
    train['dayofweek'] = temp.dayofweek
    train['hour'] = temp.hour
    train.drop(['starttime'], axis=1, inplace=True)
    print("start writing...")
    train.to_csv(w_filename, index=False)
dealDataset('E:\\Users\\jok\\PycharmProjects\\mobicup\\data\\test.csv','E:\\Users\\jok\\PycharmProjects\\mobicup\\data\\test_handler.csv')