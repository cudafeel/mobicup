import numpy as np
import re
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
'''
def hotLocation(filename,w_filename):
    train = pd.read_csv('filename')
    hotloc = train['geohashed_end_loc'].value_counts()
    hotloc_t = pd.DataFrame(hotloc)
    hotloc_t.rename(columns={'geohashed_end_loc': 'frequency'}, inplace=True)
    hotloc_t.to_csv(w_filename)
'''
def userId(filename, w_filename):
    train = pd.read_csv(filename)
    set_userid = set(train['userid'])
    userid = list(set_userid)
    print("starting")
    temp = []
    for i in range(len(userid)):
        userid_choose = train[train['userid'] == userid[i]]
        userid_choose = np.array(userid_choose)
        for j in range(len(userid_choose)):
            temp.append(list(userid_choose[j]))


    temp = pd.DataFrame(temp)
    temp.to_csv(w_filename)
userId('E:\\Users\\jok\\PycharmProjects\\mobicup\\data\\train.csv', 'E:\\Users\\jok\\PycharmProjects\\mobicup\\data\\bikeid_hotloc.csv')